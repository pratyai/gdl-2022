import torch
import numpy as np
from tsl.nn.metrics.metric_base import MaskedMetric


class MaskedQuantileLoss(MaskedMetric):
    """
        Masked quantile loss that targets a certain quantile.
        Args:
            quantile (float, optional): Must be in range (0.0, 1.0). Which quantile-point should this loss function target.
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """

    def __init__(self,
                 quantile,
                 mask_nans=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedQuantileLoss, self).__init__(metric_fn=self.loss,
                                                 mask_nans=mask_nans,
                                                 mask_inf=True,
                                                 compute_on_step=compute_on_step,
                                                 dist_sync_on_step=dist_sync_on_step,
                                                 process_group=process_group,
                                                 dist_sync_fn=dist_sync_fn,
                                                 at=at)
        self.quantile = quantile

    def loss(self, y_pred, target):
        diff = target - y_pred
        return torch.where(target > y_pred, self.quantile * diff, (self.quantile - 1) * diff)


def picp(y_hat_lower, y_hat_upper, y):
    return np.logical_and(y_hat_lower <= y, y <= y_hat_upper).astype(float).mean()


def masked_picp(y_hat_lower, y_hat_upper, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    return picp(y_hat_lower[mask], y_hat_upper[mask], y[mask])


def mpiw(y_hat_lower, y_hat_upper):
    return np.abs(y_hat_upper - y_hat_lower).mean()


def masked_mpiw(y_hat_lower, y_hat_upper, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    return mpiw(y_hat_upper, y_hat_lower)
