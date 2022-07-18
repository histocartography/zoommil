import os

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from zoommil.utils.saver import ModelSaver
from zoommil.utils.logger import MetricLogger, TBLogger
from zoommil.models.zoommil import ZoomMIL

class Trainer(object):
    def __init__(self, args, datasets):

        # general config
        self.config = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = args.max_epochs
        self.save_path = args.save_path
        
        # logger config
        self.tb_logger = TBLogger(log_dir=self.save_path)
        
        # dataloader config
        train_dataset, val_dataset, test_dataset = datasets
        if args.is_weighted_sampler:
            N = float(len(train_dataset))
            weight_per_class = [N/len(train_dataset.slide_cls_ids[c]) for c in range(len(train_dataset.slide_cls_ids))] 
            weight = [0] * int(N)
            for idx in range(len(train_dataset)):
                y = train_dataset.get_label(idx)
                weight[idx] = weight_per_class[y]
            weight = torch.DoubleTensor(weight)
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=WeightedRandomSampler(weight, len(weight)), num_workers=args.num_workers)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=RandomSampler(train_dataset), num_workers=args.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=1, sampler=SequentialSampler(val_dataset), num_workers=args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=args.num_workers)
        
        # model config
        self.n_cls = args.n_cls
        self.model = ZoomMIL(in_feat_dim=args.in_feat_dim, hidden_feat_dim=256, out_feat_dim=512, dropout=args.drop_out, 
                            k_sample=args.k_sample, k_sigma=args.k_sigma, n_cls=self.n_cls)

        # loss config
        self.ce_loss = nn.CrossEntropyLoss()

        # optimizer config
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.reg)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.scheduler_decay_rate, patience=args.scheduler_patience)
        
        # model saver config
        self.save_metric = args.save_metric
        self.model_saver = ModelSaver(save_path=self.save_path, save_metric=self.save_metric)

    def _print_model(self):
        num_params = 0
        num_params_train = 0
        print(self.model)
        
        for param in self.model.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        
        print('Total number of parameters: %d' % num_params)
        print('Total number of trainable parameters: %d' % num_params_train)

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        train_loss = 0.

        for _, data in tqdm(enumerate(self.train_loader)):
            feats = (data[0].to(self.device), data[1].to(self.device), data[2].to(self.device))
            label = data[3].to(self.device)

            # forward pass
            logits, _, _ = self.model(feats)

            # loss
            loss = self.ce_loss(logits, label)
            train_loss += loss.item()
            loss.backward()

            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss /= len(self.train_loader)
        return train_loss
    
    def _val_one_epoch(self):
        self.model.eval()
        
        val_logger = MetricLogger()
        val_loss = 0.

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.val_loader)):
                feats = (data[0].to(self.device), data[1].to(self.device), data[2].to(self.device))
                label = data[3].to(self.device)
                
                # forward pass
                logits, Y_hat, _ = self.model(feats)
                val_logger.log(Y_hat, label)
                
                # loss
                loss = self.ce_loss(logits, label)
                val_loss += loss.item()

        print('*** VALIDATION ***')
        val_summary = val_logger.get_summary()
        val_summary = {'val_{}'.format(key): val for key, val in val_summary.items()}
        val_loss /= len(self.val_loader)

        # save model ckpt
        self.model_saver(self.model, val_summary)

        return val_loss, val_summary
    
    def train(self):    
        self.model.relocate()
        self._print_model()
        
        print("Start training...")
        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch()
            val_loss, val_summary = self._val_one_epoch()
            self.scheduler.step(val_loss)
            train_summary = {
                "train_loss": train_loss, 
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=train_summary, step=epoch)
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=val_summary, step=epoch)
            
        print("Training finished!")

    def test(self):

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f'model_best_{self.save_metric}.pt')))
        self.model.relocate()
        self.model.eval()

        test_logger = MetricLogger()

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.test_loader)):
                feats = (data[0].to(self.device), data[1].to(self.device), data[2].to(self.device))
                label = data[3].to(self.device)
                
                # forward pass
                _, Y_hat, _ = self.model(feats)
                test_logger.log(Y_hat, label)
                
        print('*** TEST ***')
        test_summary = test_logger.get_summary()
        test_summary = {'test_{}'.format(key): val for key, val in test_summary.items()}
        
        self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=test_summary, step=0)
        
        # save confusion matrix
        cf_matrix = test_logger.get_confusion_matrix()
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n')
        ax.set_xlabel('\nPrediction')
        ax.set_ylabel('Ground Truth')
        ticklabels = [f'Class {i}' for i in range(self.n_cls)]
        ax.xaxis.set_ticklabels(ticklabels)
        ax.yaxis.set_ticklabels(ticklabels)
        plt.savefig(os.path.join(self.save_path, f'Confusion_Matrix.png'))
        plt.clf()
        print("Testing finished!")
        

