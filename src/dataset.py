import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from nnAudio.Spectrogram import CQT1992v2

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config


class G2NetDataset(Dataset):
    def __init__(self, df, mode="train", clip_rate=3.5):
        self.augmentation = self.__get_augmentation(mode)
        self.q_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=config.hop_length)
        self.mode = mode
        self.df = df
        self.clip_rate = clip_rate

    def __getitem__(self, idx):
        waves = np.load(self.df.loc[idx, "file_path"])
        image = self.__process_image(waves)
        image = cv2.resize(image, (config.image_size[1], config.image_size[0])) # cv2.resize(image, (width, height))
        if config.use_clip:
            mean = np.mean(image)
            std = np.std(image)
            image = np.clip(image, mean-self.clip_rate*std, mean+self.clip_rate*std)
        image = image[:, :, np.newaxis]
        image = self.__data_augmentation(self.augmentation, image)
        if self.mode == "test":
            return image
        else:
            label = torch.tensor(self.df.loc[idx, "target"]).float()
            return image, label

    def __len__(self):
        return self.df.shape[0]

    def __get_augmentation(self, mode="train"):
        if mode == "train":
            transform = [
                albu.HorizontalFlip(),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32),
                ToTensorV2(),
            ]
        else:
            transform = [
                ToTensorV2(),
            ]
        return albu.Compose(transform)

    def __data_augmentation(self, transform, image):
        augmented = transform(image=image)
        return augmented['image']

    def __process_image(self, waves):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = self.q_transform(waves)
        image = image.squeeze().numpy()
        return image