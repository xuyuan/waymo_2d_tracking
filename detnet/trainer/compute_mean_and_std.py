
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .data import TransformedDataset, ImageFolder, ConcatDataset
from .transforms.vision import ToTensor, ToRGB, Compose


datasets = [ImageFolder(p) for p in sys.argv[1:]]
dataset = ConcatDataset(datasets)
print(dataset)
dataset = TransformedDataset(dataset, Compose([ToRGB(), ToTensor(None)]))


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for sample in tqdm(loader):
        data = sample['input']
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)

mean, std = online_mean_and_sd(loader)
print(f'mean={mean}, std={std}')