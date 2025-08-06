import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import timm
import zuko
import wandb
import lightning as L

sys.path.append('../')
from utils import resnet_cond as resnet

wandb.login()


class LightningFlowDensity(L.LightningModule):
    def __init__(self, summarizer, summary_dim, 
                 cond=False, context=1, 
                 transforms=8, hidden_features=[128] * 3, 
                 lr=2e-3, 
                 scheduler=None, scheduler_epochs=100, 
                 gamma_factor=0.5, scheduler_patience=10,):
                 
        """
        Wrapper for lightning for normalizing flow trained to estimate evidence of the data. 
        
        Args:
            summarizer (nn.Module): compressor model used to produce summary statistics of the data.
            summary_dim (int): dimensionality of the summary statistics vector.
            cond (bool, optional): whether compressor is conditioned on the smoothing scale. Defaults to False.
            context (int, optional): dimensionality of the vector the flow is conditoned on. Defaults to 1 to condition on the smoothing scale.
            transforms (int, optiona): number of transforms to use in the flow transformation from the summary statistics to parameters. Defaults to 8.
            hidden_features (List, optional): number of hidden features to use in the flow transformation from the summary statistics to parameters. Defaults to (128, 128, 128).
            lr (float, optional): initial learning rate. Defaults to 2e-3.
            scheduler (str, optional): learning rate scheduler. Defaults to None.
            gamma_factor (float, optional): factor by which to reduce learning rate if using ExponentialLR or ReduceLROnPlateau scheduler. Defaults to 0.5.
            scheduler_patience (int, optional): number of epoch to wait before applying the scheduler if using ReduceLROnPlateau scheduler. Defaults to 10.
            scheduler_epochs (int, optional): maximum number of epochs to iterate the scheduler over if using CosineAnnealingLR scheduler. Defaults to 100.
            
        """
        super().__init__()
        # call this to save hyperparameters
        self.save_hyperparameters(ignore=['summarizer'])
        self.summarizer = summarizer
        self.summarizer.requires_grad_(False) 
        self.density_est = zuko.flows.MAF(features=summary_dim, 
                                          context=context, 
                                          transforms=transforms, 
                                          hidden_features=hidden_features,)
        
    def training_step(self, batch, batch_idx):
        maps, params, k = batch
        # if training density estimator for summarizer w/ k-dependence, pass k values as well
        if self.hparams.cond:
            summaries = self.summarizer([maps.float(), k.float()])
        else: 
            summaries = self.summarizer(maps.float())
            
        if self.hparams.context == 0:
            loss = -self.density_est().log_prob(summaries).mean() 
        else:
            loss = -self.density_est(k.float()).log_prob(summaries).mean()  
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True,)
        return loss
     
    def validation_step(self, batch, batch_idx):
        maps, params, k = batch
        # if training density estimator for summarizer w/ k-dependence, pass k values as well
        if self.hparams.cond:
            summaries = self.summarizer([maps.float(), k.float()])
        else: 
            summaries = self.summarizer(maps.float())
        
        if self.hparams.context == 0:
            loss = -self.density_est().log_prob(summaries).mean() 
        else:
            loss = -self.density_est(k.float()).log_prob(summaries).mean() 
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True,)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.density_est.parameters(), 
                                      lr=self.hparams.lr, 
                                      weight_decay=1e-5,)  
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max = self.hparams.scheduler_epochs, 
                                                                   eta_min=1e-6,)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        elif self.hparams.scheduler == 'plateau':        
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.hparams.gamma_factor, 
                                                                   patience=self.hparams.scheduler_patience);
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "monitor": "val_loss", "interval": "epoch"}]

        
