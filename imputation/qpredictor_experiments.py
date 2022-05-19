import copy
import datetime
import os
import pathlib
import random

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

import tsl
from tsl.data import SpatioTemporalDataModule
from tsl.data.imputation_stds import ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import (AirQuality,
                          MetrLA,
                          PemsBay)
from tsl.imputers import Imputer
from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from tsl.nn.models.imputation import (RNNImputerModel,
                                      BiRNNImputerModel,
                                      GRINModel)
from tsl.nn.utils import casting
from tsl.ops.imputation import add_missing_values
from tsl.utils import TslExperiment, ArgParser, parser_utils, numpy_metrics
from tsl.utils.neptune_utils import TslNeptuneLogger
from tsl.utils.parser_utils import str_to_bool

import qloss
from qimputer import SimpleQuantileIntervalImputer

tsl.config.config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     'config')


def get_model_class(model_str):
    if model_str == 'rnni':
        model = RNNImputerModel
    elif model_str == 'birnni':
        model = BiRNNImputerModel
    elif model_str == 'grin':
        model = GRINModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, p_fault=0., p_noise=0.):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    if dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def add_parser_arguments(parent):
    # Argument parser
    parser = ArgParser(strategy='random_search', parents=[parent],
                       add_help=False)

    # Parameters for quantile regression.
    parser.add_argument('--quantile-lower', type=float, default=0.025)
    parser.add_argument('--quantile-upper', type=float, default=0.975)

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='la_point')
    parser.add_argument("--config", type=str, default='grin.yaml')

    # Injected missing params
    parser.add_argument('--p-fault', type=float, default=0.)
    parser.add_argument('--p-noise', type=float, default=0.)

    # Splitting params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True,
                        default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batches-per-epoch', type=int, default=80)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?',
                        const=True, default=True)

    # Logging params
    parser.add_argument('--save-preds', action='store_true', default=False)
    parser.add_argument('--neptune-logger', action='store_true', default=False)
    parser.add_argument('--project-name', type=str, default="sandbox")
    parser.add_argument('--tags', type=str, default=tuple())
    # parser.add_argument('--aggregate-by', type=str, default='mean')

    known_args, _ = parser.parse_known_args()
    model_cls = get_model_class(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = Imputer.add_argparse_args(parser)
    return parser


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # tsl.logger.info(f'SEED: {args.seed}')

    model_cls = get_model_class(args.model_name)
    dataset = get_dataset(args.dataset_name, args.p_fault, args.p_noise)

    tsl.logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(tsl.config.log_dir, 'imputation',
                          args.dataset_name,
                          args.model_name,
                          exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'tsl_config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4,
                  sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    if args.model_name == 'grin':
        adj = dataset.get_connectivity(method='distance', threshold=0.1,
                                       include_self=False)
    else:
        adj = None

    # instantiate dataset
    torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      window=args.window,
                                      stride=args.stride,
                                      precision=args.precision)

    scalers = {'data': StandardScaler(axis=(0, 1))}
    splitter = dataset.get_splitter(val_len=args.val_len,
                                    test_len=args.test_len)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=args.batch_size,
        workers=args.workers
    )
    dm.setup()

    if args.in_sample:
        dm.trainset = list(range(len(torch_dataset)))

    ########################################
    # predictor                            #
    ########################################

    additional_model_hparams = dict(n_nodes=torch_dataset.n_nodes,
                                    input_size=torch_dataset.n_channels)

    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True
    )

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False)}

    # setup imputer
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    imputer_kwargs = parser_utils.filter_argparse_args(args, Imputer,
                                                       return_dict=True)

    # NOTE(pratyai): Many of these parameters should be unnecessary in the
    # context of a frozen model. How to get rid of them?
    sqii = SimpleQuantileIntervalImputer.load(
        os.path.join(
            f'../saved_model/block/{args.quantile_lower}/', 'model.pt'),
        os.path.join(
            f'../saved_model/block/{args.quantile_upper}/', 'model.pt'),
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr, 'weight_decay': args.l2_reg},
        loss_fn=MaskedMAE(compute_on_step=True),
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            'eta_min': 0.0001,
            'T_max': args.epochs
        },
        **imputer_kwargs)

    ########################################
    # logging options                      #
    ########################################

    # add tags
    tags = list(args.tags) + [args.model_name, args.dataset_name]

    if args.neptune_logger:
        logger = TslNeptuneLogger(api_key=tsl.config['neptune_token'],
                                  project_name=f"{tsl.config['neptune_username']}/{args.project_name}",
                                  experiment_name=exp_name,
                                  tags=tags,
                                  params=vars(args),
                                  offline_mode=False,
                                  upload_stdout=False)
    else:
        logger = TensorBoardLogger(
            save_dir=logdir,
            name=f'{exp_name}_{"_".join(tags)}')
    ########################################
    # testing                              #
    ########################################

    test_data = dm.test_dataloader()

    sqii.method = 'center'
    y_hat_l, y_hat_u, y_hat, y_true, mask = sqii.impute(test_data)
    res = dict(test_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.masked_mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.masked_mape(y_hat, y_true, mask),
               test_picp=qloss.masked_picp(y_hat_l, y_hat_u, y_true, mask),
               test_mpiw=qloss.masked_mpiw(y_hat_l, y_hat_u, mask))
    tsl.logger.info(('center:', res))

    sqii.method = 'uniform'
    y_hat_l, y_hat_u, y_hat, y_true, mask = sqii.impute(test_data)
    res = dict(test_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.masked_mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.masked_mape(y_hat, y_true, mask),
               test_picp=qloss.masked_picp(y_hat_l, y_hat_u, y_true, mask),
               test_mpiw=qloss.masked_mpiw(y_hat_l, y_hat_u, mask))
    tsl.logger.info(('uniform:', res))

    sqii.method = 'normal'
    y_hat_l, y_hat_u, y_hat, y_true, mask = sqii.impute(test_data)
    res = dict(test_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.masked_mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.masked_mape(y_hat, y_true, mask),
               test_picp=qloss.masked_picp(y_hat_l, y_hat_u, y_true, mask),
               test_mpiw=qloss.masked_mpiw(y_hat_l, y_hat_u, mask))
    tsl.logger.info(('normal:', res))

    if args.neptune_logger:
        logger.finalize('success')


if __name__ == '__main__':
    parser = ArgParser(add_help=False)
    parser = add_parser_arguments(parser)
    exp = TslExperiment(run_fn=run_experiment, parser=parser)
    exp.run()
