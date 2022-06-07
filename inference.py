import pandas as pd
import numpy as np
from tqdm.auto import tqdm as tqdm
import torch

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils import *
from src.models import get_model
from src.config import config


def inference(model, states, test_loader, use_tta=False):
    preds = []
    with torch.no_grad():
        for images in tqdm(test_loader, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
            images = images.to(config.device)
            avg_preds = []
            for state in states:
                model.load_state_dict(state)
                model.eval()
                if use_tta:
                    with torch.cuda.amp.autocast():
                        y_preds = model(images)
                        y_preds_tta = model(images.flip(-1)) # vertical_flip
                        # y_preds_tta_h = model(torch.flip(images, [1, 2])) # horizontal_flip
                        # y_preds_tta_t = model(torch.flip(images, [2, 3])) # transpose_flip
                    y_preds = (y_preds.sigmoid().to('cpu').numpy() + y_preds_tta.sigmoid().to('cpu').numpy()) / 2.0
                    # y_preds = (y_preds.sigmoid().to('cpu').numpy() + y_preds_tta.sigmoid().to('cpu').numpy() + y_preds_tta_h.sigmoid().to('cpu').numpy()) / 3.0
                    # y_preds = (y_preds.sigmoid().to('cpu').numpy() + y_preds_tta.sigmoid().to('cpu').numpy() + y_preds_tta_h.sigmoid().to('cpu').numpy() + y_preds_tta_t.sigmoid().to('cpu').numpy()) / 4.0
                else:
                    with torch.cuda.amp.autocast():
                        y_preds = model(images)
                    y_preds = y_preds.sigmoid().to('cpu').numpy()
                avg_preds.append(y_preds)
            avg_preds = np.mean(avg_preds, axis=0)
            preds.append(avg_preds)
        preds = np.concatenate(preds)
    return preds # -> numpy.ndarray


def create_sub(predictions, is_test=False):
    if is_test:
        df = pd.read_csv("training_labels.csv")
        df["prediction"] = predictions
        df.to_csv("train_predictions.csv", index=False)
    else:
        df = pd.read_csv("sample_submission.csv")
        df["target"] = predictions
        df[["id", "target"]].to_csv("submission.csv", index=False)



seed = 42
seed_torch(seed=seed, cudnn_benchmark=True)

do_ensemble = False
is_test = False
use_tta = False

if is_test:
    df = pd.read_csv("training_labels.csv")
else:
    df = pd.read_csv("sample_submission.csv")
print(df.shape)

if do_ensemble:
    ensemble_dict = {
        "model-1":{
            "model_name": "tf_efficientnetv2_s",
            "weight_path": "weights/2021-07-19",
            "ensemble_weight": 1,
        },
        "model-2":{
            "model_name": "tf_efficientnet_b4",
            "weight_path": "weights/2021-07-22",
            "ensemble_weight": 1,
        },
        "model-3":{
            "model_name": "tf_efficientnetv2_b3",
            "weight_path": "weights/2021-07-17",
            "ensemble_weight": 1,
        }
    }
    preds = []
    for model_no in ensemble_dict:
        config_dict = ensemble_dict[model_no]
        print(f'getting {config_dict["model_name"]}..')
        model = get_model(config_dict["model_name"], config.device, pretrained=False, num_classes=config.num_classes)
        state_dicts = [torch.load(f'{config_dict["weight_path"]}/{config_dict["model_name"]}_fold{fold}.pth') for fold in range(1, config.n_fold+1)]
        test_loader = get_dataloader(df, mode="test", batch_size=64, clip_rate=config.clip_rate)
        _predictions = inference(model, state_dicts, test_loader, use_tta)
        preds.append(_predictions)

    if ensemble_dict["model-1"]["ensemble_weight"] == 1:
        predictions = (preds[0] + preds[1]) / float(len(ensemble_dict))

    else:
        predictions = preds[0] * ensemble_dict["model-1"]["ensemble_weight"] + preds[1] * ensemble_dict["model-2"]["ensemble_weight"]# + preds[2] * ensemble_dict["model-3"]["ensemble_weight"]

    create_sub(predictions, is_test=is_test)

else:
    model = get_model(config.model_name, config.device, pretrained=False, num_classes=config.num_classes)
    state_dicts = [torch.load(f'weights/2021-08-22/{config.model_name}_fold{fold}.pth') for fold in range(1, config.n_fold+1)]
    test_loader = get_dataloader(df, mode="test", batch_size=config.batch_size*2, clip_rate=config.clip_rate)
    predictions = inference(model, state_dicts, test_loader, use_tta)
    create_sub(predictions, is_test=is_test)