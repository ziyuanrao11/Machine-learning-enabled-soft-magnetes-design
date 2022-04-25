# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:24:57 2022

@author: z.rao
"""

import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F 
from sklearn.model_selection import KFold
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import seaborn as sns

#%% read the data
all_data = pd.read_excel('all_data_Ms.xlsx').drop('Unnamed: 0',axis=1)
all_data = all_data.iloc[:,0:25]

x_all = all_data.iloc[:,0:23]
y_all = all_data['Ms']
# y_all = all_data['Hc']
x_all =x_all[y_all.isnull()==False]
y_all =y_all[y_all.isnull()==False]
#numpy to tensor
x_all = torch.FloatTensor(x_all.values) #numpy to tensor
y_all = torch.FloatTensor(y_all.values) #numpy to tensor


#to cuda
if torch.cuda.is_available():
    x_all = x_all.cuda()
    y_all = y_all.cuda() 

#%%the Neural Network
class Net(nn.Module):  
    def __init__(self, n_feature=23, n_hidden=128, n_output=1, w = 3):
        super(Net, self).__init__()    
        # self.BN=torch.nn.BatchNorm1d(n_hidden)
        self.Dropout = torch.nn.Dropout(0)
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
        nn.init.kaiming_normal_(self.hidden1.weight)
        
        self.hiddens = nn.ModuleList ([nn.Linear(n_hidden, n_hidden) for i in range(w)])                            
        for m in self.hiddens:
            nn.init.kaiming_normal_(m.weight)   
        
        self.predict = torch.nn.Linear(n_hidden, n_output)  
        nn.init.kaiming_normal_(self.predict.weight)

    def forward(self, x):  
        x = self.hidden1(x)
        # x = self.BN(x)
        x = self.Dropout(x)
        x = F.relu(x)   
        
        for m in self.hiddens:
            x = m(x)
            # x = self.BN(x)
            x = self.Dropout(x)
            x = F.relu(x) 
                      
        x = self.predict(x)
        # x = self.BN_3(x)
        # x = self.Dropout (x)
          # 输出值
        return x
    
def train(net, num_epochs, batch_size, train_features, test_features, train_labels, test_labels,
          train_loader,
          optimizer):
    print ("\n=== train begin ===")
    print(net)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for x, y in train_loader:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        if epoch % 100 == 0:
            train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
            test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
            print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
        
    print ("=== train end ===")

def test(model, test_loader):
    model.eval()
    test_loss = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss(output.view(-1, 1), target.view(-1, 1)).item()  # sum up batch loss
            n += 1
    test_loss /= n    
    print('Test set: Average loss: {:.4f}'.format(
        test_loss))
    
    return test_loss  

def plotCurve(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None,figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
    
    if legend:
        plt.legend(legend)

#%%train the NN model
    
kf = KFold(n_splits = 5) 
fold_var = 1 #start value of fivefold cross-validation
for train_index, val_index in kf.split(y_all):
    plt.close('all')
    print("fold_var=", fold_var)
    print(train_index)
    print(val_index)
    training_data = x_all[train_index]
    validation_data = x_all[val_index]
    training_y=y_all[train_index]
    validation_y=y_all[val_index]
    #numpy to tensor
    # x = torch.FloatTensor(x.values) #numpy to tensor
    # y = torch.FloatTensor(y.values) #numpy to tensor
    # training_data=torch.FloatTensor(training_data.values)
    # validation_data=torch.FloatTensor(validation_data.values)
    # training_y=torch.FloatTensor(training_y.values)
    # validation_y=torch.FloatTensor(validation_y.values)
    

    batch_size=32 
    train_dataset = Data.TensorDataset(training_data,training_y)
    test_dataset = Data.TensorDataset(validation_data, validation_y)
    
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True) 
    
    #training 
    print ("\n=== train begin ===")    
    net = Net()
    # print(net)
    if torch.cuda.is_available():
        net = net.cuda()    
    train_ls, test_ls = [], []
    loss = nn.MSELoss() 
    n_epochs = 1000
    lr=0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(n_epochs):
        for x, y in train_loader:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_ls.append(loss(net(training_data).view(-1, 1), training_y.view(-1, 1)).item())
        test_ls.append(loss(net(validation_data).view(-1, 1), validation_y.view(-1, 1)).item())
        if epoch % 100 == 0:
            print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
    print ("plot curves")
    plotCurve(range(1, n_epochs + 1), train_ls,"epoch", "loss",range(1, n_epochs + 1), 
              test_ls,["train", "validation"])
    plt.figtext(0.4, 0.75, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 15, 'color':  'black'})
    folder_dir = 'Figures'
    if not os.path.isdir(folder_dir):
      os.mkdir(folder_dir)
    fig_name_1 = 'Figures/NN_trainning_histroy_cv_Ms_{}.png'.format(fold_var)
    plt.savefig(fig_name_1, format='png', dpi=300)            
               
    #plotting
    net.eval()
    predict=net(x_all)
    if torch.cuda.is_available():
        predict=predict.cpu()
    predict=predict.data.numpy()  
    plt.figure()
    if torch.cuda.is_available():
        sns.regplot(x=predict, y=y_all.data.cpu().data.numpy(), color='g') 
    else:
        sns.regplot(x=predict, y=y_all.data.numpy(), color='g') 
    plt.xlabel('predicted')
    plt.ylabel('experimental')
    fig_name_2 = 'Figures/NN_predict_experiments_cv_Ms_{}.png'.format(fold_var)
    plt.savefig(fig_name_2, format='png', dpi=300)
    #save the models
    folder_dir = 'Results'
    if not os.path.isdir(folder_dir):
      os.mkdir(folder_dir)
    net_name = 'Results/NN_model_cv_Ms_{}.pt'.format(fold_var)
    torch.save(net.state_dict(), net_name)
    fold_var+=1
        
 