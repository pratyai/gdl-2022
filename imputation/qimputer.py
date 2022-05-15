from scipy import stats
import tsl
from tsl.imputers import Imputer
import torch
from tsl.nn.utils import casting
import numpy as np
import pytorch_lightning as pl


def impute_centerly(lower, upper):
    return (lower + upper) / 2


def impute_uniformly(lower, upper):
    width = upper - lower
    # Generate a uniform random value for each imputation point, then scale and
    # shift it appropriately.
    q = np.random.uniform(low=0.0, high=1.0, size=width.shape)
    return lower + width * q


def impute_normally(lower, upper):
    width = upper - lower
    # Generate a truncated normal random value for each imputation point, then
    # scale and shift it appropriately.
    q = stats.truncnorm.rvs(-0.5, 0.5, size=width.shape)
    return lower + width * (q + 0.5)


def load_quantile_imputer_model(model_path, **kwargs):
    imputer = Imputer(**kwargs)
    imputer.load_state_dict(torch.load(model_path))
    imputer.eval()
    imputer.freeze()
    return imputer


class SimpleQuantileIntervalImputer:
    '''
    Uses two separately trained models to estimate the lower and the upper
    bounds of the desired quantile-interval. Then yields a single imputation
    from that interval, using the desired method (which represents an implicit
    distribution within that interval).
    '''

    def __init__(self, imputer_lower, imputer_upper, method='center') -> None:
        self.imputer_lower = imputer_lower
        self.imputer_upper = imputer_upper
        self.method = method

    def load(model_path_lower, model_path_upper, method='center', **kwargs):
        lower = load_quantile_imputer_model(model_path_lower, **kwargs)
        upper = load_quantile_imputer_model(model_path_upper, **kwargs)
        return SimpleQuantileIntervalImputer(lower, upper, method)

    def impute(self, dataloader):
        trainer = pl.Trainer()
        output_lower = trainer.predict(
            self.imputer_lower, dataloaders=dataloader)
        output_lower = casting.numpy(output_lower)
        output_upper = trainer.predict(
            self.imputer_upper, dataloaders=dataloader)
        output_upper = casting.numpy(output_upper)

        y_hat_l, y_hat_u, y_true, mask = output_lower['y_hat'], output_upper[
            'y_hat'], output_lower['y'], output_lower['mask']
        # Note(pratyai): Occasionally the two predictors will predict a lower
        # bound that is slightly greater than the upper bound. This is just a
        # hack to avoid such impossible scenarios.
        y_hat_l, y_hat_u = np.where(y_hat_l <= y_hat_u, y_hat_l, y_hat_u), np.where(
            y_hat_l <= y_hat_u, y_hat_u, y_hat_l)

        imputer_fn = None
        if self.method == 'uniform':
            imputer_fn = impute_uniformly
        elif self.method == 'center':
            imputer_fn = impute_centerly
        elif self.method == 'normal':
            imputer_fn = impute_normally
        y_hat = imputer_fn(y_hat_l, y_hat_u)
        return y_hat_l, y_hat_u, y_hat, y_true, mask
