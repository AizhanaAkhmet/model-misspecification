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
class LightningFlow(L.LightningModule):
    def __init__(self, summary_dim, n_params,
                 model_name = 'resnet18',
                 transforms=8, hidden_features=[128] * 3, 
                 lr=2e-3, 
                 scheduler=None, gamma_factor=0.5, 
                 scheduler_patience=10, scheduler_epochs=100):
                 
        super().__init__()
        # call this to save hyperparameters
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.summarizer = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=summary_dim,)
        self.summarizer = replace_conv_by_circular_conv(self.summarizer)
        self.density_estimator = zuko.flows.MAF(features=n_params, 
                                                context=summary_dim, 
                                                transforms=transforms, 
                                                hidden_features=hidden_features,)
    
    # plot predicted values against true values for visual comparison
    def _log_pred_vs_true(self, batches, n_samples=(100,)):
        #batches
        x = torch.cat([batches[i][0] for i in range(len(batches))])
        y = torch.cat([batches[i][1] for i in range(len(batches))])
        k = torch.cat([batches[i][2] for i in range(len(batches))])
        
        # select the first map of every simulation in the test set
        indexes = np.arange(y.shape[0]//15)*15 + np.random.randint(0, 15)
        embedding = self.summarizer(x.float())
        predictions = self.density_estimator(embedding).sample(n_samples)
        fig, ax = plt.subplots(ncols=y.shape[-1], figsize=(5*y.shape[-1], 5))
        for i in range(y.shape[-1]):
            ax[i].plot(
                y[:,i].cpu().numpy()[indexes], 
                y[:,i].cpu().numpy()[indexes], 
                linestyle='dashed',
                color='lightgray'
            )
            ax[i].errorbar(
                y[:,i].cpu().numpy()[indexes], 
                np.mean(predictions[...,i].cpu().numpy(),axis=0)[indexes], 
                yerr=np.std(predictions[...,i].cpu().numpy(),axis=0)[indexes],  
                alpha=0.5,
                linestyle='None',
                markersize=3,
                fmt='o',
            
            )
        log_matplotlib_figure(f"true vs pred")
        plt.close()
        
    def training_step(self, batch, batch_idx):
        maps, params, k = batch
        summaries = self.summarizer(maps.float())
        loss      = -self.density_estimator(summaries).log_prob(params.float()).mean() 
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True,)
        return loss

        
    def validation_step(self, batch, batch_idx):
        maps, params, k = batch
        summaries = self.summarizer(maps.float())
        loss      = -self.density_estimator(summaries).log_prob(params.float()).mean() 
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True,)
        
        output_dict = {"batch": batch,}
        self.validation_step_outputs.append(output_dict)
               
    def on_validation_epoch_end(self):
        batches =  [self.validation_step_outputs[i]['batch'] for i in range(len(self.validation_step_outputs))]
        self._log_pred_vs_true(batches)
        self.validation_step_outputs.clear()
                    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([*self.summarizer.parameters(), 
                                       *self.density_estimator.parameters()], 
                                      lr=self.hparams.lr, 
                                      weight_decay=1e-5,)  
        if self.hparams.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma_factor,)
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "interval": "epoch"}]
        elif self.hparams.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.hparams.gamma_factor,
                                                                   patience=self.hparams.scheduler_patience,
                                                                   min_lr=1e-6,);
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "monitor": "val_loss", "interval": "epoch"}]
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max = self.hparams.scheduler_epochs, 
                                                                   eta_min=1e-6,)
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "interval": "epoch"}]
        else:
            return optimizer
        
#    
class LightningFlowCond(L.LightningModule):
    def __init__(self, summary_dim, n_params, device, field,
                 model_name='resnet18', 
                 k_cond='mlp', npe_k_cond=False,
                 transforms=8, hidden_features=[128]*3, 
                 lr=2e-3,
                 scheduler=None, gamma_factor=0.5, 
                 scheduler_patience=10,
                 scheduler_epochs=None,
                 valid_loader_array=None, valid_k=None, valid_loss_mean=None, valid_loss_std=None):
                 
        super().__init__()
        self.save_hyperparameters()
        self.field = field
        self.valid_loader_array = valid_loader_array
        self.module_device = device
        self.min_lr    = lr
        # choose which resnet and k-conditioning model to use
        if k_cond == 'mlp':
            self.summarizer = resnet.resnet_mlp(model_name=model_name, num_classes=summary_dim,)
            self.k_cond = True
        elif k_cond == 'conv_k_embedding':
            self.summarizer = resnet.resnet_conv_k_embedding(model_name=model_name, num_classes=summary_dim,) 
            self.k_cond = True
        elif k_cond == 'conv':
            self.summarizer = resnet.resnet_conv(model_name=model_name, num_classes=summary_dim,) 
            self.k_cond = True
        elif k_cond == 'conv_fc':
            self.summarizer = resnet.resnet_conv_fc(model_name=model_name, num_classes=summary_dim,) 
            self.k_cond = True
        elif k_cond == 'fc':
            self.summarizer = resnet.resnet_fc(model_name=model_name, num_classes=summary_dim,) 
            self.k_cond = True
        else:
            self.summarizer = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=summary_dim,)
            self.k_cond = False
        self.summarizer = replace_conv_by_circular_conv(self.summarizer)
        
        # set up npe (density estimator)
        self.npe_k_cond = npe_k_cond
        # whether to condition NPE on k_smoothing
        if self.npe_k_cond:
            context_dim = summary_dim + 1
        else:
            context_dim = summary_dim
        self.density_estimator = zuko.flows.MAF(features=n_params, 
                                                context=context_dim, 
                                                transforms=transforms, 
                                                hidden_features=hidden_features,)
        
    def training_step(self, batch, batch_idx):
        maps, params, k = batch
        if self.k_cond:
            summaries = self.summarizer([maps.float(), k.float()])
        else:
            summaries = self.summarizer(maps.float())
        
        if self.npe_k_cond:
            summaries    = torch.cat((summaries, k.float()), dim=1)
            
        loss      = -self.density_estimator(summaries).log_prob(params.float()).mean() 
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True,)
        return loss
     
    def validation_step(self, batch, batch_idx):
        maps, params, k = batch
        if self.k_cond:
            summaries = self.summarizer([maps.float(), k.float()])
        else:
            summaries = self.summarizer(maps.float())
            
        if self.npe_k_cond:
            summaries    = torch.cat((summaries, k.float()), dim=1)
       
        loss      = -self.density_estimator(summaries).log_prob(params.float()).mean() 
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True,)
        
    def on_validation_epoch_end(self):
        if self.valid_loader_array is not None:
            val_losses_all_scales = []
            for dataloader in self.valid_loader_array:
                val_loss, points = 0., 0
                for maps, params, k in dataloader:
                    maps    = maps.float().to(device=self.module_device)
                    params  = params.float().to(device=self.module_device)
                    k       = k.float().to(device=self.module_device)
                    bs      = maps.shape[0]
                    
                    if self.k_cond:
                        summaries = self.summarizer([maps, k])
                    else:
                        summaries = self.summarizer(maps)
                    if self.npe_k_cond:
                        summaries    = torch.cat((summaries, k), dim=1)

                    loss         = -self.density_estimator(summaries).log_prob(params).mean() 
                    val_loss     += loss*bs
                    points       += bs
                val_loss = val_loss/points
                val_losses_all_scales.append(val_loss.detach().cpu().numpy())
            self._log_compare_val_loss(val_losses_all_scales)
        
                    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([*self.summarizer.parameters(), 
                                       *self.density_estimator.parameters()], 
                                      lr=self.hparams.lr, 
                                      weight_decay=1e-5,)  
        if self.hparams.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma_factor,)
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "interval": "epoch"}]
        elif self.hparams.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.hparams.gamma_factor,
                                                                   patience=self.hparams.scheduler_patience,
                                                                   min_lr=1e-6,);
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "monitor": "val_loss", "interval": "epoch"}]
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max = self.hparams.scheduler_epochs, 
                                                                   eta_min=1e-6,)
            return [optimizer], [{"scheduler": scheduler, "frequency": 1, "interval": "epoch"}]
        else:
            return optimizer
        
    # plot comparison between validation losses of when training individual flows vs conditional flow
    def _log_compare_val_loss(self, val_losses):
        # stored values from training ensembles of individual flows
        k_scales = self.hparams.valid_k
        val_losses_no_scale_cond        = self.hparams.valid_loss_mean
        val_losses_no_scale_cond_errors = self.hparams.valid_loss_std

        
        plt.figure(figsize=(10, 6)) 
        plt.plot(k_scales, val_losses, marker='o', c='C1', label='With scale conditioning')
        plt.xscale('log')
        
        
        plt.errorbar(k_scales, 
                     val_losses_no_scale_cond, 
                     yerr=val_losses_no_scale_cond_errors,
                     elinewidth=2, capsize=4, lw=0,
                     marker='o', c='C0', label='No scale conditioning')
        
        plt.legend(loc='upper right')
        log_matplotlib_figure(f"compare validation losses")
        plt.close()
        

def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})
    
def replace_conv_by_circular_conv(model):
    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            modules_to_replace.append(name)

    for name in modules_to_replace:
        submodule = model
        name_parts = name.split(".")

        for part in name_parts[:-1]:
            submodule = getattr(submodule, part)

        module = getattr(submodule, name_parts[-1])

        if isinstance(module, nn.Conv2d):
            conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode="circular",
            )
            setattr(submodule, name_parts[-1], conv)
    return model