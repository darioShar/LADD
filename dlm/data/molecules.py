from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..graph_utils import build_edge_mask
from ..utils import print_rank_zero


@dataclass
class MoleculeMetadata:
    dataset_name: str
    max_nodes: int
    node_vocab_size: int
    edge_vocab_size: int
    node_mask_token_id: int
    edge_mask_token_id: int
    atom_encoder: dict[str, int]
    atom_decoder: dict[int, str]
    valencies: list[int]
    atom_weights: dict[int, float]
    max_weight: int
    remove_h: bool


def get_standard_splits(dataset_size: int, dataset_name: str = 'qm9', seed: int = 42):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dataset_size)

    if dataset_name == 'qm9':
        train_idx = torch.tensor(permutation[:100_000], dtype=torch.long)
        val_idx = torch.tensor(permutation[100_000:120_000], dtype=torch.long)
        test_idx = torch.tensor(permutation[120_000:], dtype=torch.long)
    elif dataset_name == 'zinc250k':
        n_val = 25_000
        n_train = dataset_size - n_val
        train_idx = torch.tensor(permutation[:n_train], dtype=torch.long)
        val_idx = torch.tensor(permutation[n_train:], dtype=torch.long)
        test_idx = val_idx
    else:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')
    return train_idx, val_idx, test_idx


def _save_processed(processed_path: Path, data: dict[str, Any]) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, processed_path)


def _load_processed(processed_path: Path) -> dict[str, Any]:
    return torch.load(processed_path, weights_only=False)


def _ensure_qm9_raw(root: Path) -> Path:
    raw_dir = root / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    required = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
    if all((raw_dir / name).exists() for name in required):
        return raw_dir

    try:
        from torch_geometric.data import download_url, extract_zip
    except ImportError as exc:
        raise ImportError('QM9 download requires torch_geometric.') from exc

    raw_url = (
        'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
        'molnet_publish/qm9.zip'
    )
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    file_path = download_url(raw_url, raw_dir)
    extract_zip(file_path, raw_dir)
    os.unlink(file_path)

    file_path = download_url(raw_url2, raw_dir)
    os.rename(raw_dir / '3195404', raw_dir / 'uncharacterized.txt')
    return raw_dir


def _ensure_qm9_splits(raw_dir: Path) -> None:
    train_path = raw_dir / 'train.csv'
    val_path = raw_dir / 'val.csv'
    test_path = raw_dir / 'test.csv'
    if train_path.exists() and val_path.exists() and test_path.exists():
        return

    import pandas as pd

    dataset = pd.read_csv(raw_dir / 'gdb9.sdf.csv')

    n_samples = len(dataset)
    n_train = 100_000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

    train.to_csv(train_path)
    val.to_csv(val_path)
    test.to_csv(test_path)


def _load_qm9_split_indices(raw_dir: Path) -> dict[str, set[int]]:
    import pandas as pd

    split_paths = {
        'train': raw_dir / 'train.csv',
        'val': raw_dir / 'val.csv',
        'test': raw_dir / 'test.csv',
    }
    splits: dict[str, set[int]] = {}
    for name, path in split_paths.items():
        df = pd.read_csv(path, index_col=0)
        splits[name] = set(df.index.tolist())
    return splits


def _process_qm9(root: Path) -> tuple[dict[str, Any], MoleculeMetadata]:
    try:
        from rdkit import Chem
        from rdkit.Chem.rdchem import BondType as BT
        from torch_geometric.utils import subgraph
    except ImportError as exc:
        raise ImportError('QM9 processing requires rdkit and torch_geometric.') from exc

    raw_dir = _ensure_qm9_raw(root)
    _ensure_qm9_splits(raw_dir)
    split_indices = _load_qm9_split_indices(raw_dir)

    split_lookup: dict[int, str] = {}
    for split_name, indices in split_indices.items():
        for idx in indices:
            split_lookup[idx] = split_name

    skip: set[int] = set()
    skip_path = raw_dir / 'uncharacterized.txt'
    if skip_path.exists():
        lines = skip_path.read_text().split('\n')[9:-2]
        skip = {int(line.split()[0]) - 1 for line in lines if line.strip()}

    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    valencies = [4, 3, 2, 1]
    atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
    max_weight = 150

    max_nodes = 9
    node_vocab_size = len(atom_encoder)
    edge_vocab_size = len(bonds) + 1
    node_mask_token_id = node_vocab_size
    edge_mask_token_id = edge_vocab_size

    nodes: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    smiles: list[str | None] = []
    num_nodes: list[int] = []
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    suppl = Chem.SDMolSupplier(str(raw_dir / 'gdb9.sdf'), removeHs=False, sanitize=False)
    for i, mol in enumerate(suppl):
        if mol is None or i in skip:
            continue
        split_name = split_lookup.get(i)
        if split_name is None:
            continue

        N = mol.GetNumAtoms()
        type_idx = [types[atom.GetSymbol()] for atom in mol.GetAtoms()]

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()] + 1]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

        if edge_index.numel() > 0:
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

        type_idx_tensor = torch.tensor(type_idx).long()
        to_keep = type_idx_tensor > 0
        edge_index, edge_attr = subgraph(
            to_keep,
            edge_index,
            edge_attr,
            relabel_nodes=True,
            num_nodes=len(to_keep),
        )
        x = x[to_keep]
        x = x[:, 1:]

        n_heavy = x.size(0)
        if n_heavy == 0 or n_heavy > max_nodes:
            continue

        node_ids = torch.full((max_nodes,), node_mask_token_id, dtype=torch.long)
        node_ids[:n_heavy] = torch.argmax(x, dim=1)

        edge_types = torch.argmax(edge_attr, dim=-1) if edge_attr.numel() > 0 else torch.empty(0, dtype=torch.long)
        edge_type_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.long)
        if edge_index.numel() > 0:
            edge_type_matrix[edge_index[0], edge_index[1]] = edge_types

        nodes.append(node_ids)
        edges.append(edge_type_matrix)
        num_nodes.append(n_heavy)

        try:
            mol_copy = Chem.Mol(mol)
            mol_copy = Chem.RemoveHs(mol_copy)
            Chem.SanitizeMol(mol_copy)
            smiles.append(Chem.MolToSmiles(mol_copy, canonical=True))
        except Exception:
            smiles.append(None)

        dataset_idx = len(nodes) - 1
        if split_name == 'train':
            train_idx.append(dataset_idx)
        elif split_name == 'val':
            val_idx.append(dataset_idx)
        else:
            test_idx.append(dataset_idx)

    nodes_tensor = torch.stack(nodes, dim=0)
    edges_tensor = torch.stack(edges, dim=0)
    num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.long)

    metadata = MoleculeMetadata(
        dataset_name='qm9',
        max_nodes=max_nodes,
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        node_mask_token_id=node_mask_token_id,
        edge_mask_token_id=edge_mask_token_id,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        valencies=valencies,
        atom_weights=atom_weights,
        max_weight=max_weight,
        remove_h=True,
    )

    data_dict = {
        'nodes': nodes_tensor,
        'edges': edges_tensor,
        'num_nodes': num_nodes_tensor,
        'smiles': smiles,
        'splits': {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        },
        'metadata': metadata,
    }
    return data_dict, metadata


def _process_zinc250k(root: Path) -> tuple[dict[str, Any], MoleculeMetadata]:
    try:
        import pandas as pd
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError('ZINC250k processing requires pandas and rdkit.') from exc

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / 'zinc250k.csv'
    if not csv_path.exists():
        import urllib.request

        url = (
            'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/'
            'zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
        )
        print_rank_zero('Downloading ZINC250k...')
        urllib.request.urlretrieve(url, csv_path)

    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].tolist()

    atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    max_nodes = 38
    node_mask_token_id = len(atom_encoder)
    edge_vocab_size = 4
    edge_mask_token_id = edge_vocab_size

    bond_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
    }

    nodes = []
    edges = []
    smiles = []
    num_nodes = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_nodes:
            continue

        node_ids = torch.full((max_nodes,), node_mask_token_id, dtype=torch.long)
        valid = True
        for idx, atom in enumerate(mol.GetAtoms()):
            sym = atom.GetSymbol()
            if sym not in atom_encoder:
                valid = False
                break
            node_ids[idx] = atom_encoder[sym]
        if not valid:
            continue

        edge_type = torch.zeros((max_nodes, max_nodes), dtype=torch.long)
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_id = bond_map.get(bond.GetBondType())
            if bond_id is None:
                continue
            edge_type[u, v] = bond_id
            edge_type[v, u] = bond_id

        nodes.append(node_ids)
        edges.append(edge_type)
        num_nodes.append(num_atoms)
        smiles.append(smi)

    nodes_tensor = torch.stack(nodes, dim=0)
    edges_tensor = torch.stack(edges, dim=0)
    num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.long)

    metadata = MoleculeMetadata(
        dataset_name='zinc250k',
        max_nodes=max_nodes,
        node_vocab_size=len(atom_encoder),
        edge_vocab_size=edge_vocab_size,
        node_mask_token_id=node_mask_token_id,
        edge_mask_token_id=edge_mask_token_id,
        atom_encoder=atom_encoder,
        atom_decoder=atom_decoder,
        valencies=[4, 3, 2, 1, 5, 6, 1, 1, 1],
        atom_weights={0: 12, 1: 14, 2: 16, 3: 19, 4: 31, 5: 32, 6: 35.5, 7: 80, 8: 127},
        max_weight=1000,
        remove_h=True,
    )

    data_dict = {
        'nodes': nodes_tensor,
        'edges': edges_tensor,
        'num_nodes': num_nodes_tensor,
        'smiles': smiles,
        'metadata': metadata,
    }
    return data_dict, metadata


def load_molecule_dataset(
    data_dir: str,
    dataset_name: str,
    regenerate: bool = False,
) -> tuple[dict[str, Any], MoleculeMetadata]:
    root = Path(data_dir).expanduser().resolve() / dataset_name
    processed_path = root / 'processed.pt'

    if (not regenerate) and processed_path.exists():
        data = _load_processed(processed_path)
        metadata = data['metadata']
        if not hasattr(metadata, 'valencies'):
            if metadata.dataset_name == 'qm9':
                metadata.valencies = [4, 3, 2, 1]
                metadata.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
                metadata.max_weight = 150
                metadata.remove_h = True
            elif metadata.dataset_name == 'zinc250k':
                metadata.valencies = [4, 3, 2, 1, 5, 6, 1, 1, 1]
                metadata.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 31, 5: 32, 6: 35.5, 7: 80, 8: 127}
                metadata.max_weight = 1000
                metadata.remove_h = True
        return data, metadata

    if dataset_name == 'qm9':
        data, metadata = _process_qm9(root)
    elif dataset_name == 'zinc250k':
        data, metadata = _process_zinc250k(root)
    else:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')

    _save_processed(processed_path, data)
    meta_path = root / 'metadata.json'
    meta_path.write_text(json.dumps(metadata.__dict__, indent=2))
    return data, metadata


class MoleculeDataset(Dataset):
    def __init__(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        metadata: MoleculeMetadata,
        indices: torch.Tensor,
        smiles: list[str] | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.indices = indices
        self.metadata = metadata
        self.smiles = smiles or []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self.indices[idx].item()
        node_ids = self.nodes[real_idx]
        edge_ids = self.edges[real_idx]
        node_mask = node_ids != self.metadata.node_mask_token_id
        edge_mask = build_edge_mask(node_mask, exclude_diagonal=True)

        item = {
            'nodes': node_ids,
            'edges': edge_ids,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
        }
        if self.smiles:
            item['smiles'] = self.smiles[real_idx]
        return item


class MoleculeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        batch_size: int = 256,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
        regenerate: bool = False,
        split_seed: int = 42,
        return_smiles: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.regenerate = regenerate
        self.split_seed = split_seed
        self.return_smiles = return_smiles
        self.metadata: MoleculeMetadata | None = None

        self.datasets: dict[str, MoleculeDataset] = {}

    def prepare_data(self) -> None:
        load_molecule_dataset(self.data_dir, self.dataset_name, regenerate=self.regenerate)

    def setup(self, stage: str | None = None) -> None:
        data, metadata = load_molecule_dataset(self.data_dir, self.dataset_name, regenerate=self.regenerate)
        self.metadata = metadata
        nodes = data['nodes']
        edges = data['edges']
        smiles = data.get('smiles', []) if self.return_smiles else None
        splits = data.get('splits')
        if splits is not None:
            train_idx = torch.tensor(splits['train'], dtype=torch.long)
            val_idx = torch.tensor(splits['val'], dtype=torch.long)
            test_idx = torch.tensor(splits['test'], dtype=torch.long)
        else:
            train_idx, val_idx, test_idx = get_standard_splits(len(nodes), self.dataset_name, seed=self.split_seed)

        self.datasets['train'] = MoleculeDataset(nodes, edges, metadata, train_idx, smiles=smiles)
        self.datasets['val'] = MoleculeDataset(nodes, edges, metadata, val_idx, smiles=smiles)
        self.datasets['test'] = MoleculeDataset(nodes, edges, metadata, test_idx, smiles=smiles)
        self._compute_dataset_distributions(nodes, edges, train_idx, val_idx)

    def _compute_dataset_distributions(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> None:
        if self.metadata is None:
            return
        node_mask_token_id = self.metadata.node_mask_token_id
        edge_mask_token_id = self.metadata.edge_mask_token_id
        max_nodes = self.metadata.max_nodes
        node_vocab_size = self.metadata.node_vocab_size
        edge_vocab_size = self.metadata.edge_vocab_size

        train_nodes = nodes[train_idx]
        train_edges = edges[train_idx]
        val_nodes = nodes[val_idx]

        all_nodes = torch.cat((train_nodes, val_nodes), dim=0)
        node_mask_all = all_nodes != node_mask_token_id
        n_nodes = node_mask_all.sum(dim=1)
        n_dist = torch.bincount(n_nodes, minlength=max_nodes + 1).float()
        if n_dist.sum() > 0:
            n_dist = n_dist / n_dist.sum()

        node_mask_train = train_nodes != node_mask_token_id
        node_ids = train_nodes[node_mask_train]
        node_type_dist = torch.bincount(node_ids, minlength=node_vocab_size).float()
        if node_type_dist.sum() > 0:
            node_type_dist = node_type_dist / node_type_dist.sum()

        edges_clean = train_edges.clone()
        edges_clean[edges_clean == edge_mask_token_id] = 0
        pair_mask = build_edge_mask(node_mask_train, exclude_diagonal=True)
        edge_types = edges_clean[pair_mask]
        edge_type_dist = torch.bincount(edge_types, minlength=edge_vocab_size).float()
        if edge_type_dist.sum() > 0:
            edge_type_dist = edge_type_dist / edge_type_dist.sum()

        bond_orders = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5], device=edges_clean.device)
        bond_vals = bond_orders[edges_clean.clamp(max=bond_orders.numel() - 1)]
        bond_vals = bond_vals * pair_mask.float()
        valency = bond_vals.sum(dim=2)
        valency_flat = valency[node_mask_train]
        valency_idx = valency_flat.long().clamp(max=3 * max_nodes - 3)
        valency_dist = torch.bincount(valency_idx, minlength=3 * max_nodes - 2).float()
        if valency_dist.sum() > 0:
            valency_dist = valency_dist / valency_dist.sum()

        self.n_nodes_dist = n_dist
        self.node_type_dist = node_type_dist
        self.edge_type_dist = edge_type_dist
        self.valency_dist = valency_dist

    @staticmethod
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = torch.stack([item['nodes'] for item in batch], dim=0)
        edges = torch.stack([item['edges'] for item in batch], dim=0)
        node_mask = torch.stack([item['node_mask'] for item in batch], dim=0)
        edge_mask = torch.stack([item['edge_mask'] for item in batch], dim=0)
        result = {
            'nodes': nodes,
            'edges': edges,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
        }
        if 'smiles' in batch[0]:
            result['smiles'] = [item['smiles'] for item in batch]
        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['val'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate,
        )
