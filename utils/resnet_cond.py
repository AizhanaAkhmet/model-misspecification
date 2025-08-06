from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import timm


class resnet_conv(nn.Module):
    """ResNet-based model which accounts for the smoothing scale by adding it as an additional channel."""
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
