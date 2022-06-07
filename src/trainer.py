import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm as tqdm

from sklearn.metrics import roc_auc_score

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config as config
from src.utils import mixup_data, get_dataloader
from src.metric_recorder import MetricRecorder
from src.loss import epoch_update_gamma


class Trainer:
    def __init__(self, model, dataloaders_dict, optimizer, criterion, scheduler=None, logger=None, do_mixup=False):
        # 初期値設定
        self.best_loss = 10**5
        self.best_score = 0.0
        self.counter = 0 # early_stopまでのカウンター
        self.early_stop_limit = 5 # スコア更新の失敗の回数でearly stopの実施を決める
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.n_fold = int()

        # setter
        self.model = model
        self.train_loader = dataloaders_dict["train"]
        self.valid_loader = dataloaders_dict["val"]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        if logger is None:
            raise ValueError("logger is NoneType.")
        else:
            self.logger = logger

        self.freezed = True if config.freeze else False
        if self.freezed:
            self.logger.info("Currently the model is frozen.")
        self.do_mixup = do_mixup
        self.is_model_save = True
        self.save_result_image = True

        self.metric_recorder = MetricRecorder(config.epochs)

    def set_n_fold(self, n_fold):
        self.n_fold = n_fold

    # train
    def fit(self):
        epoch_gamma = None
        last_whole_y_pred = None
        last_whole_y_t = None

        for epoch in range(config.epochs):
            self.logger.info(f"Epoch {epoch+1} / {config.epochs}")

            # 4epochから解除 config.freeze = Trueでのみ適用
            if self.freezed and epoch + 1 == 4:
                self.logger.info(f"unfreeze model..")
                self._unfreeze_model()
                self.freezed = False

            # ラスト5epochでmixupなしにする
            if self.do_mixup and epoch + 1 == config.epochs - 5:
                self.logger.info(f"start learning without mixup augmentation..")
                self._off_mixup()

            for phase in ["train", "val"]:
                if phase == "train":
                    if self.do_mixup:
                        epoch_loss = self._train_with_mixup()
                    elif config.use_roc_star:
                        epoch_loss, whole_y_pred, whole_y_t = self._train_with_roc_star(epoch, epoch_gamma, last_whole_y_t, last_whole_y_pred)
                    else:
                        epoch_loss = self._train()
                    self.metric_recorder.record_score_when_training(epoch_loss)
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}')
                else:
                    if config.use_roc_star:
                        epoch_loss, epoch_score, epoch_gamma, last_whole_y_t, last_whole_y_pred = self._valid_with_roc_star(epoch, whole_y_pred, whole_y_t)
                    else:
                        epoch_loss, epoch_score = self._valid()
                    self.metric_recorder.record_score_when_validating(epoch_loss, epoch_score)
                    self.logger.info(f'fold - {self.n_fold}:: phase: {phase}, loss: {epoch_loss:.4f}, roc_auc: {epoch_score:.4f}, -- learning rate: {self.optimizer.param_groups[0]["lr"]}')
                    self._update_score(epoch_loss, epoch_score)

            if config.early_stop and self.counter > self.early_stop_limit:
                self.logger.debug("early stopping..")
                break

        if self.save_result_image:
            self.metric_recorder.save_as_image(file_name=f"fold_{self.n_fold}.png")

        return self.best_score


    def _train(self):
        self.model.train()
        epoch_loss = 0.0

        for _, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]
            self.optimizer.zero_grad()

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.view(-1), labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        return epoch_loss


    def _train_with_mixup(self):
        self.model.train()
        epoch_loss = 0.0

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config.alpha)
                    inputs, targets_a, targets_b = inputs.to(config.device), targets_a.to(config.device), targets_b.to(config.device)
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1)
                    loss = self.criterion(outputs, targets_a) * lam + self.criterion(outputs, targets_b) * (1. - lam)

                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (idx + 1) % config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
            else:
                raise NotImplementedError("try use amp = True")

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        return epoch_loss


    def _train_with_roc_star(self, epoch, epoch_gamma=None, last_whole_y_t=None, last_whole_y_pred=None):
        self.model.train()
        epoch_loss = 0.0
        whole_y_pred = np.array([])
        whole_y_t = np.array([])
        criterion = torch.nn.BCEWithLogitsLoss()

        for _, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            batch_size = inputs.shape[0]
            self.optimizer.zero_grad()

            if config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    if epoch == 0:
                        loss = criterion(outputs.view(-1), labels)
                    else:
                        outputs = outputs.sigmoid()
                        loss = self.criterion(outputs.view(-1), labels, epoch_gamma, last_whole_y_t, last_whole_y_pred)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            else:
                outputs = self.model(inputs)
                if epoch == 0:
                    loss = criterion(outputs.view(-1), labels)
                else:
                    loss = self.criterion(outputs.view(-1), labels, epoch_gamma, last_whole_y_t, last_whole_y_pred)
                loss.backward()
                self.optimizer.step()

            if torch.isnan(loss):
                raise ValueError("contains the loss value of nan")

            whole_y_pred = np.append(whole_y_pred, outputs.clone().detach().cpu().numpy())
            whole_y_t    = np.append(whole_y_t, labels.clone().detach().cpu().numpy())

            epoch_loss += loss.item() * batch_size
            
        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        return epoch_loss, whole_y_pred, whole_y_t


    def _valid(self):
        self.model.eval()
        epoch_loss, epoch_score = float(), float()
        y_pred, y_true = [], []

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.valid_loader):
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                batch_size = inputs.shape[0]

                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs.view(-1), labels)

                if torch.isnan(loss):
                    raise ValueError("contains the loss value of nan")

                epoch_loss += loss.item() * batch_size

                y_pred.extend(outputs.sigmoid().to("cpu").numpy())
                y_true.extend(labels.to("cpu").numpy())

        epoch_loss = epoch_loss / len(self.valid_loader.dataset)
        epoch_score = roc_auc_score(y_true, y_pred)

        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, epoch_score

    
    def _valid_with_roc_star(self, epoch, whole_y_pred, whole_y_t):
        self.model.eval()
        epoch_loss, epoch_score = float(), float()
        y_pred, y_true = [], []
        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.valid_loader):
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                batch_size = inputs.shape[0]

                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = criterion(outputs.view(-1), labels)
                if torch.isnan(loss):
                    raise ValueError("contains the loss value of nan")

                epoch_loss += loss.item() * batch_size

                y_pred.extend(outputs.sigmoid().to("cpu").numpy())
                y_true.extend(labels.to("cpu").numpy())

        epoch_loss = epoch_loss / len(self.valid_loader.dataset)
        epoch_score = roc_auc_score(y_true, y_pred)

        if self.scheduler is not None:
            self.scheduler.step()

        last_whole_y_t = torch.tensor(whole_y_t).cuda()
        last_whole_y_pred = torch.tensor(whole_y_pred).cuda()
        epoch_gamma = epoch_update_gamma(last_whole_y_t, last_whole_y_pred, epoch)

        return epoch_loss, epoch_score, epoch_gamma, last_whole_y_t, last_whole_y_pred


    def _update_score(self, epoch_loss, epoch_score):
        if self.best_score <= epoch_score:
            self.best_score = epoch_score
            self.best_loss = epoch_loss
            self.logger.debug(f"update best score: {self.best_score:.4f}")

            if self.is_model_save:
                torch.save(self.model.state_dict(), f"{config.weight_path}/{config.model_name}_fold{self.n_fold}.pth")
            self.counter = 0
        
        elif self.best_loss >= epoch_loss:
            self.best_loss = epoch_loss
            self.counter = 0

        else:
            self.logger.debug("There is no update of the best score")
            self.counter += 1


    def _unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # 残り5epochのときにmixupなしで学習する用
    def _off_mixup(self):
        self.do_mixup = False


    def not_save_model(self):
        self.is_model_save = False


    def not_save_result_image(self):
        self.save_result_image = False