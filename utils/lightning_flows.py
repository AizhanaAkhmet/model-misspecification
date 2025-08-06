import sys, io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import timm
import zuko
import lightning as L
import wandb

sys.path.append('../')
from utils import resnet_cond as resnet

wandb.login()


class LightningFlow(L.LightningModule):
    def __init__(self, summary_dim, n_params,
                 model_name = 'resnet18',
                 transforms=8, hidden_features=[128] * 3, 
                 lr=2e-3, 
                 scheduler=None, gamma_factor=0.5, 
                 scheduler_patience=10, scheduler_epochs=100):
        """
        Wrapper for lightning for model without conditioning on smoothing scale.
        
        Args:
            summary_dim (int): dimensionality of the summary statistics vector.
            n_params (int): number of parameters to predict.
            model_name (str, optional): name of the model from timm repository to use as a compressor. Defaults to 'resnet18'.
            transforms (int, optiona): number of transforms to use in the flow transformation from the summary statistics to parameters. Defaults to 8.
            hidden_features (List, optional): number of hidden features to use in the flow transformation from the summary statistics to parameters. Defaults to (128, 128, 128).
            lr (float, optional): initial learning rate. Defaults to 2e-3.
            scheduler (str, optional): learning rate scheduler. Defaults to None.
            gamma_factor (float, optional): factor by which to reduce learning rate if using ExponentialLR or ReduceLROnPlateau scheduler. Defaults to 0.5.
            scheduler_patience (int, optional): number of epoch to wait before applying the scheduler if using ReduceLROnPlateau scheduler. Defaults to 10.
            scheduler_epochs (int, optional): maximum number of epochs to iterate the scheduler over if using CosineAnnealingLR scheduler. Defaults to 100.
        """
                 
        super().__init__()
        
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
        

class LightningFlowCond(L.LightningModule):
    def __init__(self, summary_dim, n_params, device, 
                 model_name='resnet18', 
                 k_cond='conv', npe_k_cond=False,
                 transforms=8, hidden_features=[128]*3, 
                 lr=2e-3,
                 scheduler=None, gamma_factor=0.5, 
                 scheduler_patience=10,
                 scheduler_epochs=None,
                 valid_loader_array=None, valid_k=None, valid_loss_mean=None, valid_loss_std=None):
        """
        Wrapper for lightning for model with conditioning on Gaussian smoothing scale.
        
        Args:
            summary_dim (int): dimensionality of the summary statistics vector.
            n_params (int): number of parameters to predict.
            device (str): module device. 
            model_name (str, optional): name of the model from timm repository to use as a compressor. Defaults to 'resnet18'.
            k_cond (str, optional): type of conditioning on smoothing scale used by the compressor network. Defaults to 'conv'.
            npe_k_cond (bool, optional): whether to condition the density estimator on smoothing scale. Defaults to False.
            transforms (int, optiona): number of transforms to use in the flow transformation from the summary statistics to parameters. Defaults to 8.
            hidden_features (List, optional): number of hidden features to use in the flow transformation from the summary statistics to parameters. Defaults to (128, 128, 128).
            lr (float, optional): initial learning rate. Defaults to 2e-3.
            scheduler (str, optional): learning rate scheduler. Defaults to None.
            gamma_factor (float, optional): factor by which to reduce learning rate if using ExponentialLR or ReduceLROnPlateau scheduler. Defaults to 0.5.
            scheduler_patience (int, optional): number of epoch to wait before applying the scheduler if using ReduceLROnPlateau scheduler. Defaults to 10.
            scheduler_epochs (int, optional): maximum number of epochs to iterate the scheduler over if using CosineAnnealingLR scheduler. Defaults to 100.
        """         
        super().__init__()
        self.save_hyperparameters()
        self.valid_loader_array = valid_loader_array
        self.module_device = device
        self.min_lr    = lr
        # choose which resnet and k-conditioning model to use
        if k_cond == 'conv':
            self.summarizer = resnet.resnet_conv(model_name=model_name, num_classes=summary_dim,) 
            self.k_cond = True
        else:
            self.summarizer = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=summary_dim,)
            self.k_cond = False
            
        # replace convolutions by circular convolutions
        self.summarizer = replace_conv_by_circular_conv(self.summarizer)
        
        # Set up npe density estimator
        # and choose whether to condition NPE on k_smoothing
        self.npe_k_cond = npe_k_cond
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
        
        

def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer
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
    """Replaces convolutions in a model with circular convolutions."""
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