import random
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

# local files
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config
from src.dataset import G2NetDataset


#Pytorchで再現性を保つ
def seed_torch(seed=42, cudnn_benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark


def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)
    
    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = FileHandler(log_folder)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger


def get_dataloaders_dict(train_df, val_df, clip_rate=3.5):
    train_dataset = G2NetDataset(train_df, mode="train", clip_rate=clip_rate)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True)
    val_dataset = G2NetDataset(val_df, mode="val", clip_rate=clip_rate)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)

    return {"train": train_dataloader, "val": val_dataloader}


def get_dataloader(df, mode="train", batch_size=64, clip_rate=3.5):
    dataset = G2NetDataset(df, mode=mode, clip_rate=clip_rate)
    shuffle = True if mode == "train" else False
    drop_last = True if mode == "train" else False
    data_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=config.num_workers, drop_last=drop_last)
    
    return data_loader


def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data