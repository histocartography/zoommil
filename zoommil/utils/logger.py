import importlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

class TBLogger(object):
    def __init__(self, log_dir=None):
        super(TBLogger, self).__init__()
        self.log_dir = log_dir
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)
    
    def end(self):
        self.tb_logger.flush()
        self.tb_logger.close()
    
    def run(self, func_name, *args, mode="tb", **kwargs):
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)
        return None

    def tb_log_scalars(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)

class MetricLogger(object):
    def __init__(self):
        super(MetricLogger, self).__init__()
        self.y_pred = []
        self.y_true = []

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.y_pred.append(Y_hat)
        self.y_true.append(Y)

    def get_summary(self):
        acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred) # accuracy
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=None) # f1 score
        weighted_f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average='weighted') # weighted f1 score
        kappa = cohen_kappa_score(y1=self.y_true, y2=self.y_pred, weights='quadratic') # cohen's kappa

        print('*** Metrics ***')
        print('* Accuracy: {}'.format(acc))
        for i in range(len(f1)):
            print('* Class {} f1-score: {}'.format(i, f1[i]))
        print('* Weighted f1-score: {}'.format(weighted_f1))
        print('* Kappa score: {}'.format(kappa))
        
        summary = {'accuracy': acc, 'weighted_f1': weighted_f1,'kappa': kappa}
        for i in range(len(f1)):
            summary[f'class_{i}_f1'] = f1[i]
        return summary

    def get_confusion_matrix(self):
        cf_matrix = confusion_matrix(np.array(self.y_true), np.array(self.y_pred)) # confusion matrix
        return cf_matrix








