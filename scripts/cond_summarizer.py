import os, sys
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
import wandb
wandb.login()

sys.path.append('../')
from utils import datasets as datasets
import utils.lightning_flows as LFlows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: %s'%(device))

# source code for the progress bar: https://github.com/Lightning-AI/pytorch-lightning/issues/15283
class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
    
# parse training arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_name",
                    help="name of the run",
                    required=True,)

parser.add_argument("--project_name",
                    help="name of the project",
                    required=True,)

parser.add_argument("--field",
                    help="type of the astrophysical field",
                    required=True,)

parser.add_argument("--suite",
                    help="name of the suite",
                    required=True,)

parser.add_argument("--home_dir",
                    default="/n/netscratch/dvorkin_lab/Lab/aakhmetzhanova/evidence-estimation-Astrid/",
                    help="home_directory",
                    required=False,)

parser.add_argument("--save_dir",
                    default="/n/netscratch/dvorkin_lab/Lab/aakhmetzhanova/evidence-estimation-Astrid/trained_models/",
                    help="output_directory (within home directory)",
                    required=False,)

parser.add_argument("--n_params", 
                    default=6,
                    help="number of parameters to estimate", 
                    type=int,
                    required=False,)

parser.add_argument("--splits", 
                    default=15,
                    help="number of maps per simulation to use", 
                    type=int,
                    required=False,)

parser.add_argument("--k_min", 
                    default=2.,
                    help="minimum smoothing scale", 
                    type=float,
                    required=False,)

parser.add_argument("--k_max", 
                    default=45.,
                    help="maximum smoothing scale", 
                    type=float,
                    required=False,)

parser.add_argument("--normalize_k", action="store_true", 
                    help="whether to normalize smoothing scales during sampling")

parser.add_argument("--k_cond", 
                    default=None,
                    help="whether to condition the summaries on the smoothing scale and in which way")

parser.add_argument("--npe_k_cond", action="store_true", 
                    help="whether to condition the density estimator on the smoothing scale")

parser.add_argument("--linear_sampling", action="store_true", 
                    help="whether to sample smoothing scales linearly")

parser.add_argument("--lr", 
                    help="learning rate", 
                    default=5e-4,
                    type=float,
                    required=False,)

parser.add_argument("--max_epochs", 
                    default=300,
                    help="maximum epochs to train the model for", 
                    type=int,
                    required=False,)

parser.add_argument("--batch_size", 
                    default=100,
                    help="training/validation batch size", 
                    type=int,
                    required=False,)

parser.add_argument("--seed", 
                    default=1,
                    help="random seed for splitting the data set into training/validation/test sets", 
                    type=int,
                    required=False,)

parser.add_argument("--summary_dim",
                    default=40,
                    type=int,
                    help="dimensionality of the learned summary statistic",
                    required=False,)

parser.add_argument( "--n_transforms",
                    default=8,
                    type=int,
                    help="number of flow transforms",
                    required=False,)

parser.add_argument( "--n_hidden_f",
                    default=256,
                    type=int,
                    help="number of hidden features",
                    required=False,)

parser.add_argument( "--n_hidden_l",
                    default=3,
                    type=int,
                    help="number of layers of hidden features",
                    required=False,)

parser.add_argument('--scheduler', 
                    help='learning rate scheduler',
                    default=None,
                    required=False,)

parser.add_argument("--gamma_factor", 
                    help="factor by which to reduce the learning rate with ReduceLROnPlateau scheduler", 
                    default=0.5,
                    type=float,
                    required=False,)

parser.add_argument("--scheduler_patience", 
                    help="patience argument for the ReduceLROnPlateau scheduler", 
                    default=10,
                    type=float,
                    required=False,)

parser.add_argument("--model_name",
                    default='resnet18',
                    help="name of the architecture",
                    required=False,)

parser.add_argument("--early_stopping", action="store_true", 
                    help="whether to use the early stopping callback")

args = parser.parse_args()
## ARGUMENT PARSER ##

# load data
home_dir    = args.home_dir

splits      = args.splits
n_params    = args.n_params

params  = np.loadtxt(home_dir+'data/params_LH_{:s}.txt'.format(args.suite),)[:, :n_params]
minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[:n_params]
maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[:n_params]
params  = (params - (minimum+maximum)/2)/((maximum - minimum)/2)   # rescale parameters
params  = np.repeat(params[:, None, :], splits, axis = 1)          # reshape the parameters to match the shape of the maps 
params  = torch.tensor(params).float()

grid     = 256
BoxSize  = 25

maps     = np.load(home_dir+ 'data/2D/Maps_{:s}_{:s}_LH_z=0.00.npy'.format(args.field, args.suite))
if args.field=='Mstar':
    maps = maps+1.0 # add 1 to avoid NaNs if looking at the stellar mass density fields
    
maps     = maps.reshape(params.shape[0], -1, 1, grid, grid)[:, :splits]
maps_mean, maps_std = np.log10(maps).mean(), np.log10(maps).std()
maps     = torch.tensor(maps).float()


# directory to save trained models
save_dir     = args.save_dir + 'Maps_{:s}/'.format(args.field)
os.makedirs(save_dir, exist_ok=True)

# training hyperparameters
max_epochs  = args.max_epochs
lr          = args.lr
summary_dim = args.summary_dim

transforms  = args.n_transforms
hidden_features =[args.n_hidden_f] * args.n_hidden_l

batch_size   = args.batch_size
seed         = args.seed
k_min, k_max = args.k_min, args.k_max

train_frac, valid_frac, test_frac = 0.9, 0.05, 0.05
train_dset, valid_dset, test_dset = datasets.create_datasets_maps(maps, params, 
                                                                  train_frac, valid_frac, test_frac,
                                                                  seed=seed, rotations=True,
                                                                  smoothing=True, 
                                                                  k_min=k_min, k_max=k_max,
                                                                  normalize_k=args.normalize_k,
                                                                  linear=args.linear_sampling, 
                                                                  log_scale=True, 
                                                                  standardize=True, 
                                                                  maps_mean=maps_mean, maps_std=maps_std,) 
train_loader = DataLoader(train_dset,  batch_size, shuffle = True, )
valid_loader = DataLoader(valid_dset,  batch_size, shuffle = False, )
test_loader  = DataLoader(test_dset,   batch_size, shuffle = False, )

# set up callbacks
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', filename='best_val_loss', save_last=True)
early_callback = EarlyStopping(monitor="val_loss", mode="min", patience=40,)
callbacks=[checkpoint_callback, lr_monitor, MyProgressBar()]  

    
# set up logger
wandb_logger = WandbLogger(project=args.project_name, 
                           save_dir=save_dir,
                           log_model=False, 
                           version=args.run_name,
                           name=args.run_name,) 

if args.early_stopping:
    callbacks.append(early_callback)
    
# set up trainer
trainer = Trainer(accelerator='gpu', 
                  logger=wandb_logger, 
                  max_epochs=max_epochs, 
                  callbacks=callbacks)

# initialize trainer
flow_net = LFlows.LightningFlowCond(summary_dim, n_params, device, 
                                    args.field,
                                    k_cond=args.k_cond, npe_k_cond=args.npe_k_cond,
                                    lr=lr, transforms=transforms, hidden_features=hidden_features,
                                    model_name=args.model_name,
                                    scheduler=args.scheduler, 
                                    scheduler_epochs=max_epochs,)
trainer.fit(model=flow_net, train_dataloaders=train_loader, val_dataloaders=valid_loader,)    
wandb.finish()
