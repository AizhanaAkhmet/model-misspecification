import os, sys, io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import timm
import zuko

import lightning as L
sys.path.append('/n/home11/aakhmetzhanova/evidence-estimation/')
from utils import resnet_cond as resnet

import wandb
wandb.login(key='981c0ae085ae974bf46691d143fccf689859be3d')

#
class LightningFlowDensity(L.LightningModule):
    def __init__(self, summarizer, summary_dim, 
                 cond=False, context=0, 
                 transforms=8, hidden_features=[128] * 3, 
                 lr=2e-3, 
                 scheduler=None, scheduler_epochs=600, 
                 gamma_factor=0.5, scheduler_patience=10,):
                 
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

        
