import numpy as np
import torch
from torchmetrics import Metric


# to compute W_p loss
def compute_mean_lploss(tens, lploss=2.0):
    # simply flatten the tensor
    return torch.pow(
        torch.linalg.norm(tens.flatten(), ord=lploss),
        # dim = list(range(0, len(tens.shape)))),
        lploss,
    ) / torch.prod(torch.tensor(tens.shape), dim=0)


def wasserstein_1d_torch(proj_data, proj_model, trim):
    proj_data = torch.sort(proj_data, dim=-1).values
    proj_model = torch.sort(proj_model, dim=-1).values
    n = min(proj_data.shape[1], proj_model.shape[1])
    k = int(trim * n)
    if k > 0:
        # remove 1% smallest and largest values to avoid outliers
        proj_data = proj_data[:, k : n - k]
        proj_model = proj_model[:, k : n - k]
    return torch.mean(torch.abs(proj_data - proj_model))


def compute_sliced_wasserstein(
    data: torch.Tensor,
    model_data: torch.Tensor,
    n_projections: int = 1000,
) -> float:
    # Ensure the data is on CPU for numpy-based Wasserstein, if your function relies on numpy
    data = data.float()
    model_data = model_data.float()

    N_data, c, d = data.shape
    N_model, _, d_model = model_data.shape

    assert d == d_model, 'Data and model data must have the same dimension'

    data = data.view(N_data, -1)
    model_data = model_data.view(N_model, -1)

    # directions = torch.rand(d, n_projections, device=data.device) / d
    # directions = torch.rand(d - 1, n_projections, device=data.device)
    # # add 0 and 1 to direction
    # directions = torch.cat([torch.zeros(1, n_projections), directions, torch.ones(1, n_projections)], dim=0)
    # directions = torch.sort(directions, dim=0).values
    # directions = directions[1:] - directions[:-1]

    # Sample unit directions: columns have ||u||_2 = 1
    directions = torch.randn(d, n_projections, device=data.device)
    directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-9)

    # proj_data = ((2 * data - 1) @ directions).T
    # proj_model = ((2 * model_data - 1) @ directions).T
    proj_data = (data @ directions).T
    proj_model = (model_data @ directions).T

    sw = wasserstein_1d_torch(proj_data, proj_model, trim=0.0)
    return sw.item()


class SlicedWassersteinMetric(Metric):
    """PyTorch Lightning metric for computing sliced Wasserstein distance.

    Computes the sliced Wasserstein distance between real and generated data
    by projecting high-dimensional data onto random 1D projections and
    computing the Wasserstein distance in each projection.
    """

    def __init__(self, n_projections: int = 1000, pad_token_id: int | None = None):
        super().__init__()
        self.n_projections = n_projections
        self.pad_token_id = pad_token_id

        # Store accumulated data for batch computation
        self.add_state('real_data', default=[], dist_reduce_fx=None)
        self.add_state('generated_data', default=[], dist_reduce_fx=None)

    def _pad_to_width(self, tensor: torch.Tensor, target_width: int) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError(f'SlicedWassersteinMetric expects rank-2 tensors, got shape {tuple(tensor.shape)}.')
        current_width = int(tensor.size(1))
        if current_width == target_width:
            return tensor
        if current_width > target_width:
            return tensor[:, :target_width]

        pad_value = 0.0 if self.pad_token_id is None else float(self.pad_token_id)
        pad = torch.full(
            (tensor.size(0), target_width - current_width),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, pad), dim=1)

    def _pad_batches_to_shared_width(self, batches: list[torch.Tensor]) -> list[torch.Tensor]:
        if not batches:
            return batches
        target_width = max(int(batch.size(1)) for batch in batches)
        return [self._pad_to_width(batch, target_width) for batch in batches]

    def update(self, real_data: torch.Tensor, generated_data: torch.Tensor):
        """Update the metric state with real and generated data.

        Args:
            real_data: Real data tensor [batch_size, sequence_length]
            generated_data: Generated data tensor [batch_size, sequence_length]
        """
        if real_data.ndim != 2 or generated_data.ndim != 2:
            raise ValueError(
                'SlicedWassersteinMetric expects 2D [batch_size, sequence_length] tensors for both inputs.',
            )
        # Keep original shape [N, D] for proper sliced Wasserstein computation
        self.real_data.append(real_data.float())
        self.generated_data.append(generated_data.float())

    def compute(self):
        """Compute the sliced Wasserstein distance.

        Returns:
            Sliced Wasserstein distance between real and generated data
        """
        if not self.real_data or not self.generated_data:
            return torch.tensor(0.0)

        padded_real = self._pad_batches_to_shared_width(list(self.real_data))
        padded_generated = self._pad_batches_to_shared_width(list(self.generated_data))
        target_width = max(
            max(int(batch.size(1)) for batch in padded_real),
            max(int(batch.size(1)) for batch in padded_generated),
        )
        padded_real = [self._pad_to_width(batch, target_width) for batch in padded_real]
        padded_generated = [self._pad_to_width(batch, target_width) for batch in padded_generated]

        # Concatenate all accumulated data maintaining [N, D] shape
        all_real = torch.cat(padded_real, dim=0)  # [N_real, D]
        all_generated = torch.cat(padded_generated, dim=0)  # [N_gen, D]

        # Reshape to [N, c, D] format expected by compute_sliced_wasserstein
        real_reshaped = all_real.unsqueeze(1)  # [N_real, 1, D]
        gen_reshaped = all_generated.unsqueeze(1)  # [N_gen, 1, D]

        try:
            sw_distance = compute_sliced_wasserstein(
                real_reshaped,
                gen_reshaped,
                n_projections=self.n_projections,
            )
            return torch.tensor(sw_distance, dtype=torch.float32)
        except Exception:
            # Fallback to simple L1 distance if sliced Wasserstein fails
            return torch.mean(torch.abs(all_real.mean(dim=0) - all_generated.mean(dim=0)))


# updated to manually compute bins if requested
def compute_wasserstein_distance(
    data,
    gen_samples,
    manual_compute=False,
    num_samples=-1,
    distance='euclidean',
    normalized=True,
    bins='auto',
    _range=None,
):
    import pyemd

    if manual_compute:
        # each vector is a histogram fof bins
        # that is the density of the corresponding data in each bin.
        # pairwise distance is computed between each bin
        # thus if each data point has its own bin, we need 2*N bins
        # and a 4*N*N matrix.
        # use L1 loss
        lploss = 1.0
        N = np.min((data.shape[0], gen_samples.shape[0])) if num_samples == -1 else num_samples
        # equal histograms, first bins for data, second one for gen_samples
        data_1_array = np.concatenate((np.ones(N) / N, np.zeros(N)), dtype=np.float64)
        data_2_array = np.concatenate((np.zeros(N), np.ones(N) / N), dtype=np.float64)
        distance_data = np.zeros((2 * N, 2 * N), dtype=np.float64)
        for i in range(2 * N):
            for j in range(2 * N):
                # distance_data = np.array([[compute_loss_all(data1[i] - data2[j]) for j in range(data2.shape[0])] for i in range(data1.shape[0])])
                data_i = data[i] if i < N else gen_samples[i % N]
                data_j = data[j] if j < N else gen_samples[j % N]
                distance_data[i, j] = compute_mean_lploss(data_i - data_j, lploss=lploss)
        res = pyemd.emd(data_1_array, data_2_array, np.float64(distance_data))
        res = res ** (lploss)
        return res
    return pyemd.emd_samples(
        gen_samples[:num_samples],
        data[:num_samples],
        distance=distance,
        normalized=normalized,
        bins=bins,
        range=_range,
    )
