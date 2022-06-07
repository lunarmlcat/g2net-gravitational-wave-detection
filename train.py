import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
import warnings
warnings.filterwarnings("ignore")

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import *
from src.models import get_model, get_custom_model
from src.config import config
from src.loss import ROC_Star
from src.trainer import Trainer


if config.do_mixup and config.use_roc_star:
    raise ValueError("cannot start learning with do_mixup = True and use_roc_star = True")

if not os.path.exists(config.weight_path):
    os.makedirs(config.weight_path, exist_ok=True)
    
if not os.path.exists("result"):
    os.mkdir("result")

cv_score = []

train_df = pd.read_csv("training_labels.csv")
print(train_df.shape)
logger = setup_logger(config.log_file_name)
seed_torch(cudnn_benchmark=True)


for fold in range(config.n_fold):
    logger.info(f"fold #{fold+1}")
    
    _train_df = train_df.query(f"kfold != {fold}").reset_index(drop=True)
    _val_df = train_df.query(f"kfold == {fold}").reset_index(drop=True)
    
    logger.debug(f"getting {config.model_name}.. device: {config.device}")
    model = get_model(config.model_name, config.device, pretrained=True, num_classes=config.num_classes, model_freeze=config.freeze)
    # model = get_custom_model(config.model_name, config.device, pretranied=True, num_classes=config.num_classes, model_freeze=config.freeze)
    dataloaders_dict = get_dataloaders_dict(_train_df, _val_df, clip_rate=config.clip_rate)
    if config.use_roc_star:
        criterion = ROC_Star()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    T_mult = 1
    eta_min = 1e-8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs, T_mult=T_mult, eta_min=eta_min)
    
    trainer = Trainer(model, dataloaders_dict, optimizer, criterion, scheduler=scheduler, logger=logger, do_mixup=config.do_mixup)
    trainer.set_n_fold(fold+1)
    best_score = trainer.fit()
    logger.info(f"fold #{fold+1} best score: {best_score:.4}")
    cv_score.append(best_score)

logger.debug(cv_score)

# mlflow
with mlflow.start_run():
    params = {
        "input_size": config.image_size,
        "hop_length": config.hop_length,
        "freeze_model": config.freeze,
        "use_clip": config.use_clip,
        "clip_rate": config.clip_rate,
        "mixup": config.do_mixup,
        "model_name": config.model_name,
        "batch_size": config.batch_size,
        "use_roc_star": config.use_roc_star,
        "loss_function": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": config.lr,
        "scheduler": scheduler.__class__.__name__,
        "scheduler T_0": config.epochs,
        "scheduler T_mult": T_mult,
        "scheduler eta_min": eta_min,
        "n_fold": config.n_fold,
        "epochs": config.epochs,
        "early_stop": config.early_stop,
        "remarks": "baseline",
    }
    mlflow.log_params(params)

    for idx, score in enumerate(cv_score):
        mlflow.log_metric(f"fold {idx+1}", score)
        mlflow.log_artifact(f"result/fold_{idx+1}.png")

    cv_score = np.mean(cv_score, axis=0)
    logger.debug(f"cv score: {cv_score:.4}")
    mlflow.log_metric("cv score", cv_score)

mlflow.end_run()