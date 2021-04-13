import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from copy import deepcopy
import time
import torch.nn.functional as F

class dataset(data.Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

class datasetTest(data.Dataset):

    def __init__(self, data_x):
        self.data_x = data_x

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index]

class Perceptron(nn.Module):
    def __init__(self, input_shape, params, classifier=False):
        super(Perceptron, self).__init__()
        
        self.params = params
            
        self.layer1 = nn.Linear(input_shape, input_shape//2)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(input_shape//2, input_shape//4)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.2)

        self.out = nn.Linear(input_shape//4, 1)

    
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x1 = self.dp1(self.relu1(self.layer1(x-0.5)))
        x2 = self.dp2(self.relu2(self.layer2(x1)))
        
        xout = self.out(x2)
            
        return xout

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class NeuralNet:
    def __init__(self, max_epochs=1000, batch_size=1024, classifier=False):
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.classifier = classifier
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_params(self, **params):
        self.params = params
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, train_x, train_y, eval_metric, eval_set, verbose=-1, early_stopping_rounds=None):
        model = Perceptron(train_x.shape[1], self.params, classifier=self.classifier)
        model.to(self.device)
        
        train_dataset = dataset(train_x, train_y)
        val_dataset = dataset(eval_set[0], eval_set[1])

        if not self.classifier:
            criterion = nn.MSELoss().to(self.device)
        else:
            criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

        val_scores = []
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        best_epoch = -1
        best_val_loss = 1e+308
        best_val_score = -1e+308

        for epoch in range(self.epochs):

            if get_lr(optimizer) <= 1e-6:
                break

            model.train()            
            
            train_epoch_loss, epoch_count = 0.0, 0
            for i, data in enumerate(train_loader):
                x, y = data[0].float().reshape(data[0].shape[0],1,1,-1).to(self.device), data[1].float().reshape(-1, 1).to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)
                train_epoch_loss += loss.item()*x.shape[0]
                epoch_count += x.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_epoch_loss /= epoch_count
            
            model.eval()
            val_y, val_preds = [], []
            val_loss = 0.0
            val_cnt = 0.0
            for i, data in enumerate(val_loader):
                x, y = data[0].float().reshape(data[0].shape[0],1,1,-1).to(self.device), data[1].float().reshape(-1, 1).to(self.device)
                outputs = model(x)
                #print(y, outputs)
                val_loss += criterion(outputs, y)*y.shape[0]
                val_cnt += y.shape[0]

                if self.classifier:
                    outputs = torch.sigmoid(outputs)                
                    outputs = (outputs>0.5).long()
                val_y += list(y.flatten().cpu().numpy())
                val_preds += list(outputs.flatten().detach().cpu().numpy())
            
            val_loss /= val_cnt

            val_y = np.array(val_y)
            val_preds = np.array(val_preds)

            if self.classifier:
                val_score = eval_metric(val_y.astype(np.int32), val_preds.astype(np.int32))[1]
            else:
                val_score = eval_metric(val_y.astype(np.float32), val_preds.astype(np.float32))[1]
                
            val_scores.append(val_score)
            if val_score > best_val_score:
                best_val_score = val_score
                self.model = deepcopy(model).to(self.device)
            print (epoch, train_epoch_loss, val_loss.item(), val_score, get_lr(optimizer))
            scheduler.step(val_score)
            
    def predict(self, val_x):
        self.model.to(self.device)
        self.model.eval()
        val_preds = []
        
        if val_x.shape[0] > 1:
            val_dataset = datasetTest(val_x)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            for i, data in enumerate(val_loader):
                x = data.float().reshape(data.shape[0],1,1,-1).to(self.device)
                outputs = self.model(x)
                if self.classifier:
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs>0.5).long()
                val_preds += list(outputs.flatten().detach().cpu().numpy())
        else:
            x = torch.Tensor(val_x).float().reshape(val_x.shape[0],1,1,-1).to(self.device)
            outputs = self.model(x)
            if self.classifier:
                outputs = torch.sigmoid(outputs)
                outputs = (outputs>0.5).long()

            val_preds = [outputs.flatten().detach().cpu().numpy()]
        #print(val_preds)
        return np.array(val_preds).flatten()

    