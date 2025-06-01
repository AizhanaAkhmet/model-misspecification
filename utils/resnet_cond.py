from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

import timm

#######################################
class resnet_mlp(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer = timm.create_model(model_name=model_name, pretrained=False, in_chans=1, num_classes=512,)
        
        ## MLP for including smoothing scale 
        self.k_fc1   = nn.Linear(512 + 1, 512)
        self.k_relu1 = nn.ReLU(inplace=True)
        self.k_fc2   = nn.Linear(512, 512)
        self.k_relu2 = nn.ReLU(inplace=True)
        self.k_fc3   = nn.Linear(512, num_classes)
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        x, k = l
        x = self.summarizer(x)
        x = torch.cat((x, k), dim=1)
        
        # add smoothing scale k
        ## embedd smoothing scale k ##
        x = self.k_fc1(x)
        x = self.k_relu1(x)
        x = self.k_fc2(x)
        x = self.k_relu2(x)
        x = self.k_fc3(x)
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)

#######################################
class resnet_fc(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer    = timm.create_model(model_name=model_name, pretrained=False, in_chans=1, num_classes=num_classes,)
        fc_in_features     = self.summarizer.fc.in_features
        self.summarizer.fc = nn.Linear(fc_in_features + 1, num_classes, bias=True)
        
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        x, k = l
        
        x = self.summarizer.forward_features(x)
        x = self.summarizer.global_pool(x)
        x = torch.cat((x, k), dim=1)
        x = self.summarizer.fc(x)
        
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)

#######################################
class resnet_conv_k_embedding(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer = timm.create_model(model_name=model_name, pretrained=False, in_chans=1, num_classes=num_classes,)
        
        ## MLP for including smoothing scale 
        self.k_fc1   = nn.Linear(1, 512)
        self.k_relu1 = nn.ReLU(inplace=True)
        self.k_fc2   = nn.Linear(512, 512)
        self.k_relu2 = nn.ReLU(inplace=True)
        self.k_fc3   = nn.Linear(512, 64)
        
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        x, k = l
        
        ## embedd the smoothing scale
        k_emb = self.k_fc1(k)
        k_emb = self.k_relu1(k_emb)
        k_emb = self.k_fc2(k_emb)
        k_emb = self.k_relu2(k_emb)
        k_emb = self.k_fc3(k_emb)
        x = self.summarizer.conv1(x)
        
        ## add k-embedding after the first convolutional layer ##
        x = x + k_emb[:, :, None, None]

        # continue with the regular ResNet layers
        x = self.summarizer.bn1(x)
        x = self.summarizer.act1(x)
        x = self.summarizer.maxpool(x)

        x = self.summarizer.layer1(x)
        x = self.summarizer.layer2(x)
        x = self.summarizer.layer3(x)
        x = self.summarizer.layer4(x)
        
        # apply the fully-connected laters
        x = self.summarizer.global_pool(x)
        x = self.summarizer.fc(x)
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)

#######################################    
class resnet_conv(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer = timm.create_model(model_name=model_name, pretrained=False, in_chans=2, num_classes=num_classes,)
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        x, k = l
        # add k scale as an additional channel
        k = k[:, :, None, None]
        k = k.repeat(1, 1, x.shape[-1], x.shape[-1])
        x = torch.cat((x, k), dim=1) 
        
        # continue with the subsequent resnet layers
        x = self.summarizer.conv1(x)
        x = self.summarizer.bn1(x)
        x = self.summarizer.act1(x)
        x = self.summarizer.maxpool(x)

        x = self.summarizer.layer1(x)
        x = self.summarizer.layer2(x)
        x = self.summarizer.layer3(x)
        x = self.summarizer.layer4(x)
        
        # apply the fully-connected laters
        x = self.summarizer.global_pool(x)
        x = self.summarizer.fc(x)
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)

#######################################    
class resnet_conv_fc(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer = timm.create_model(model_name=model_name, pretrained=False, in_chans=2, num_classes=num_classes,)
        fc_in_features     = self.summarizer.fc.in_features
        self.summarizer.fc = nn.Linear(fc_in_features + 1, num_classes, bias=True)
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        x, k = l
        # add k scale as an additional channel
        k = k[:, :, None, None]
        k = k.repeat(1, 1, x.shape[-1], x.shape[-1])
        x = torch.cat((x, k), dim=1) 
        
        # continue with the subsequent resnet layers
        x = self.summarizer.conv1(x)
        x = self.summarizer.bn1(x)
        x = self.summarizer.act1(x)
        x = self.summarizer.maxpool(x)

        x = self.summarizer.layer1(x)
        x = self.summarizer.layer2(x)
        x = self.summarizer.layer3(x)
        x = self.summarizer.layer4(x)
        
        # apply the fully-connected laters
        x = self.summarizer.global_pool(x)
        x = torch.cat((x, k[:, :, 0, 0]), dim=1)
        x = self.summarizer.fc(x)
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)

#######################################
class resnet_conv_layers(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        add_k_embedding: bool = False,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.summarizer = timm.create_model(model_name=model_name, 
                                            pretrained=False, in_chans=1, 
                                            num_classes=num_classes,)
        
        ## MLP for including smoothing scale 
        self.k_fc1   = nn.Linear(1, 512)
        self.k_relu1 = nn.ReLU(inplace=True)
        self.k_fc2   = nn.Linear(512, 512)
        self.k_relu2 = nn.ReLU(inplace=True)
        self.k_fc3   = nn.Linear(512, 64)
        
        
        self.k_fc4   = nn.Linear(64, 128)
        self.k_relu3 = nn.ReLU(inplace=True)
        
        self.k_fc5   = nn.Linear(128, 256)
        self.k_relu4 = nn.ReLU(inplace=True)
        
        self.k_fc6   = nn.Linear(256, 512)
        self.k_relu5 = nn.ReLU(inplace=True)
        
        self.k_fc7   = nn.Linear(512, 512)
        
    def _forward_impl(self, l: List[Tensor]) -> Tensor:
        # See note [TorchScript super()]
        x, k = l
    
        ## embedd the smoothing scale
        k_emb = self.k_fc1(k)
        k_emb = self.k_relu1(k_emb)
        k_emb = self.k_fc2(k_emb)
        k_emb = self.k_relu2(k_emb)
        k_emb = self.k_fc3(k_emb)
        
        x = self.summarizer.conv1(x)
        
        ## add k-embedding after the first convolutional layer ##
        x = x + k_emb[:, :, None, None]
        
        # continue with the regular ResNet layers
        x = self.summarizer.bn1(x)
        x = self.summarizer.act1(x)
        x = self.summarizer.maxpool(x)
        
        
        x = self.summarizer.layer1(x)
        x = x + k_emb[:, :, None, None]
        
        k_emb = self.k_fc4(k_emb)
        k_emb = self.k_relu3(k_emb)
        x = self.summarizer.layer2(x)
        x = x + k_emb[:, :, None, None]
        
        
        k_emb = self.k_fc5(k_emb)
        k_emb = self.k_relu4(k_emb)
        x = self.summarizer.layer3(x)
        x = x + k_emb[:, :, None, None]
        
        
        k_emb = self.k_fc6(k_emb)
        k_emb = self.k_relu5(k_emb)
        x = self.summarizer.layer4(x)
        x = x + k_emb[:, :, None, None]
        
        
        # apply the fully-connected laters
        k_emb = self.k_fc7(k_emb)
        x = self.summarizer.global_pool(x)
        x = x + k_emb
        
        x = self.summarizer.fc(x)
        
        return x
        
    def forward(self, l: List[Tensor]) -> Tensor:
        return self._forward_impl(l)