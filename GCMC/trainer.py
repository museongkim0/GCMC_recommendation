import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self,cfg, model, dataset, data, calc_rmse, print_pred,
                 optimizer, experiment=None):
        self.feature = cfg.feature
        self.model = model
        self.dataset = dataset
        self.data = data
        self.calc_rmse = calc_rmse
        self.optimizer = optimizer
        self.experiment = experiment
        self.print_pred = print_pred

    def training(self, epochs):
        self.epochs = epochs
        train_rmse_list =[]
        test_rmse_list = []
        for epoch in range(self.epochs):
            loss, train_rmse = self.train_one(epoch)
            test_rmse = self.test()
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
            self.summary(epoch, loss, train_rmse, test_rmse)
            if self.experiment is not None:
                metrics = {'loss': loss,
                           'train_rmse': train_rmse,
                           'test_rmse': test_rmse}
                self.experiment.log_metrics(metrics, step=epoch)

        # 결과 rmse 시각화 저장
        plt.figure(figsize=(7,7))
        plt.plot(train_rmse_list, color = 'b', label = 'Train')
        plt.plot(test_rmse_list, color = 'g', label = 'Test')
        plt.legend(loc = 'best')
        plt.title(self.feature)
        plt.grid(True)
        #plt.savefig('/content/drive/MyDrive/gcmc_colab2/data/book/raw/Test.png', dpi = 100)
        plt.savefig('/content/drive/MyDrive/Test_{}.png'.format(self.feature))
        
        print('END TRAINING')

    def train_one(self, epoch):
        self.model.train()
        out = self.model(self.data.x, self.data.edge_index,
                         self.data.edge_type, self.data.edge_norm)
        loss = F.cross_entropy(out[self.data.train_idx], self.data.train_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        #self.print_pred(out[self.data.test_idx], self.data.test_gt)
        return loss.item(), rmse.item()

    def test(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index, 
                         self.data.edge_type, self.data.edge_norm)
        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)
        return rmse.item()

    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch+1, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch+1, self.epochs, loss, train_rmse, test_rmse))
