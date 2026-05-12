import torch
from torchmetrics import Metric


class StandardDeviationMetric(Metric):
    """PyTorch Lightning metric for computing standard deviation.

    Computes the standard deviation of values across batches,
    properly handling distributed aggregation.
    """

    def __init__(self):
        super().__init__()
        self.add_state('sum_values', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_squared_values', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, values: torch.Tensor):
        """Update the metric state with new values.

        Args:
            values: Tensor of values to compute std from
        """

        values = values.float()
        B, D = values.shape

        # Lazy init states with correct shape/device
        if len(self.sum_values.shape) < 2:
            device = values.device
            self.sum_values = torch.zeros(D, device=device)
            self.sum_squared_values = torch.zeros(D, device=device)

        # Accumulate per-component sums across the batch (dim=0)
        self.sum_values += values.sum(dim=0)
        self.sum_squared_values += (values * values).sum(dim=0)
        self.count += B

        # # Flatten and convert to float
        # flat_values = values.flatten().float()

        # # Update accumulators
        # self.sum_values += torch.sum(flat_values)
        # self.sum_squared_values += torch.sum(flat_values**2)
        # self.count += flat_values.numel()

    def compute(self):
        """Compute the standard deviation.

        Returns:
            Standard deviation of all accumulated values
        """
        n = int(self.count.item())
        if n == 0:
            # pick a sensible device
            dev = self.sum_values.device if self.sum_values.numel() > 0 else None
            return torch.tensor(0.0, device=dev)

        n_t = self.count.to(self.sum_values.dtype)

        mean = self.sum_values / n_t
        mean_sq = self.sum_squared_values / n_t
        var = mean_sq - mean**2
        var = var * (n_t / (n_t - 1))
        var = torch.clamp(var, min=0.0)
        std_per_comp = torch.sqrt(var)
        return std_per_comp.mean()

        # if self.count == 0:
        #     return torch.tensor(0.0)

        # Compute mean
        # mean = self.sum_values / self.count

        # # Unbiased variance calculation
        # mean_squared = self.sum_squared_values / self.count
        # variance = (mean_squared - (mean**2)) * (self.count / (self.count - 1))

        # # Handle numerical precision issues
        # variance = torch.clamp(variance, min=0.0)

        # return torch.sqrt(variance)


class MinMetric(Metric):
    """PyTorch Lightning metric for computing minimum value."""

    def __init__(self):
        super().__init__()
        self.add_state('min_value', default=torch.tensor(float('inf')), dist_reduce_fx='min')

    def update(self, values: torch.Tensor):
        """Update the metric state with new values."""
        min_val = torch.min(values)
        self.min_value = torch.min(self.min_value, min_val)

    def compute(self):
        """Compute the minimum value."""
        return self.min_value if self.min_value != float('inf') else torch.tensor(0.0)


class MaxMetric(Metric):
    """PyTorch Lightning metric for computing maximum value."""

    def __init__(self):
        super().__init__()
        self.add_state('max_value', default=torch.tensor(float('-inf')), dist_reduce_fx='max')

    def update(self, values: torch.Tensor):
        """Update the metric state with new values."""
        max_val = torch.max(values)
        self.max_value = torch.max(self.max_value, max_val)

    def compute(self):
        """Compute the maximum value."""
        return self.max_value if self.max_value != float('-inf') else torch.tensor(0.0)


class MeanMetric(Metric):
    """PyTorch Lightning metric for computing mean value."""

    def __init__(self):
        super().__init__()
        self.add_state('sum_values', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, values: torch.Tensor):
        """Update the metric state with new values."""
        flat_values = values.flatten().float()
        self.sum_values += torch.sum(flat_values)
        self.count += flat_values.numel()

    def compute(self):
        """Compute the mean value."""
        if self.count == 0:
            return torch.tensor(0.0)
        return self.sum_values / self.count
