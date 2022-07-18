import os
import torch
import numpy as np

class ModelSaver:
    def __init__(self, save_path, save_metric='loss'):
        """
        Args:
            save_path (str): Path to save the model
            save_metric (str, optional): Save metric. Defaults to 'loss'.
        """
        self.save_metric = save_metric
        self.save_path = save_path 
        self.best_loss = np.inf
        self.best_f1 = 0.

    def __call__(self, model, summary):
        if self.save_metric == 'loss':
            if summary["val_loss"] < self.best_loss:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {summary["val_loss"]:.6f}).  Saving model ...')
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_loss.pt"))
                self.best_loss = summary["val_loss"]
        elif self.save_metric == 'f1':
            if summary["val_weighted_f1"] > self.best_f1:
                print(f'Validation weighted f1 increased ({self.best_f1:.6f} --> {summary["val_weighted_f1"]:.6f}).  Saving model ...')
                torch.save(model.state_dict(), os.path.join(self.save_path, "model_best_f1.pt"))
                self.best_f1 = summary["val_weighted_f1"]
        else:
            raise NotImplementedError