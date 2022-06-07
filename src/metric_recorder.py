import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MetricRecorder():
    def __init__(self, epochs):
        self.train_losses = list()
        self.train_scores = list()
        self.val_losses = list()
        self.val_scores = list()

        self.epochs = epochs


    def record_score_when_training(self, train_loss, train_score=None):
        self.train_losses.append(train_loss)
        if train_score is not None:
            self.train_scores.append(train_score)


    def record_score_when_validating(self, val_loss, val_score):
        self.val_losses.append(val_loss)
        self.val_scores.append(val_score)


    def save_as_image(self, file_name):
        start, end = 1, len(self.train_losses)+1
        step_size = 1
        epochs = [val for val in range(1, len(self.train_losses)+1)]

        _, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].plot(epochs, self.train_losses, color="blue", linestyle="solid", label="train loss")
        ax[0].plot(epochs, self.val_losses, color="green", linestyle="solid", label="val loss")
        ax[0].set_title(f"losses")

        ax[0].xaxis.set_ticks(np.arange(start, end, step_size))
        ax[0].legend()
        ax[0].grid(True)

        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("roc_auc")
        # TODO: train scores
        ax[1].plot(epochs, self.val_scores, color="orange", linestyle="solid", label="val roc auc")
        ax[1].set_title(f"roc auc")

        ax[1].xaxis.set_ticks(np.arange(start, end, step_size))
        ax[1].legend()
        ax[1].grid(True)

        plt.savefig(f"result/{file_name}")
        plt.close()