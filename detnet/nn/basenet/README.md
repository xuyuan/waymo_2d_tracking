

# ImageNet pretrained models

| model              | origin               | FLOPS | No. Param | Top 5 Acc | Top 1 Acc |
|:-------------------|:---------------------|:------|:----------|:----------|:----------|
| vgg11              | torchvision          |       |           |           |           |
| resnet18           | torchvision          |       |           |     89.08 |     69.76 |
| resnet50           | torchvision          |       |           |     92.97 |     76.15 |
| resnet101          | torchvision          |       |           |     93.56 |     77.37 |
| efficientnet-b0    | efficientnet_pytorch |       |           |           |           |
| se_resnet50        | pretrainedmodels     |       |           |           |           |
| se_resnext50_32x4d | pretrainedmodels     |       |           |           |           |
| resnext50_32x4d    | torchvision          |       |           |           |           |  
| resnext101_32x4d   | pretrainedmodels     |       |           |           |           |
| resnext101_32x8d   | torchvision          |       |           |           |           |
| resnext101_64x4d   | pretrainedmodels     |       |           |           |           |
| resnext101_32x8d   | WSL-Images           |  16B  |       88M |      96.4 |      82.2 |
| resnext101_32x16d  | WSL-Images           |  36B  |      193M |      97.2 |      84.2 |
| resnext101_32x32d  | WSL-Images           |  87B  |      466M |      97.5 |      85.1 |
| resnext101_32x48d  | WSL-Images           | 153B  |      829M |      97.6 |      85.4 |
| resnet50_ibn_a     | IBN-Net              |       |           |     93.59 |     77.24 |
| resnet101_ibn_a    | IBN-Net              |       |           |     94.41 |     78.61 |
| densenet121        | torchvision          |       |           |     92.17 |     74.65 |