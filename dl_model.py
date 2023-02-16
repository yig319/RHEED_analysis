import os, glob, random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


def prepare_dataloader(h5_para_file, growth_dict, batch_size, visualize=False):
    spot = 'spot_2'
    metric = 'img_intensity'
    fit_settings={'fit_func': 'exp', 'I_diff': 13000, 'bounds':[0.01, 1], 'p_init':(1, 0.1),
                  'from_previous': False, 'weight_previous': False, 'relative_intensity': False, 'use_prev_as_bound': True}


    length = 500
    a, tau, x, info = analyze_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, interval=1000,  
                                     visualize=False, fit_settings=fit_settings)
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all, I_drop_list_all = info
    y_pca = denoise_data(xs_all, ys_nor_all, visualize=visualize)

    full_dataset = torch.utils.data.TensorDataset(torch.tensor(y_pca).unsqueeze(1))
    dl = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    return dl, y_pca


class model_linear(nn.Module):
    def __init__(self):
        super().__init__()   

        self.bn1 = nn.BatchNorm1d(1)
        self.linear1 = nn.Linear(500, 1000) 
        
        self.bn2 = nn.BatchNorm1d(1)
        self.linear2 = nn.Linear(1000, 2000)
        
        self.bn3 = nn.BatchNorm1d(1)
        self.linear3 = nn.Linear(2000, 2000)
        
        self.bn4 = nn.BatchNorm1d(1)
        self.linear4 = nn.Linear(2000, 100)
        
        self.linear5 = nn.Linear(100, 9)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = self.linear2(x)
        x = self.relu(x)        
        
        x = self.bn3(x)
        x = self.linear3(x)
        x = self.relu(x)
        
        x = self.bn4(x)
        x = self.linear4(x)
        x = self.relu(x)
        
        x = self.linear5(x)
        x = self.relu(x)

        a, b, c = x[:, 0, 0], x[:, 0, 1], x[:, 0, 2]
        m1, m2, relax_12 = x[:, 0, 3], x[:, 0, 4], x[:, 0, 5]
        m3, m4, relax_34 = x[:, 0, 6], x[:, 0, 7], x[:, 0, 8]
        
        t = torch.linspace(0, 1, 500).unsqueeze(1).repeat(1, len(x)).to(torch.float32).to(x.device)
        y_fit = a + b*t + c*(t**2) + (m1 + m2*t)*(1-torch.exp(-relax_12*10 * t)) + (m3 + m4*t)*(1-torch.exp(-relax_34*10 * t))
        y_fit = torch.swapaxes(y_fit.unsqueeze(1), 0, 2)
        return y_fit, x

def train(model, train_dl, epochs, start, device, loss_func, optimizer, scheduler=None):
    model.train()
    train_data_size = len(train_dl.dataset)

    for epoch_idx in range(start, epochs+start):
        print("Epoch: {}/{}".format(epoch_idx+1, epochs+start))

        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_dl)):
            inputs = batch[0]
            inputs = inputs.to(torch.float32).to(device)
            model = model.to(device)

            optimizer.zero_grad()
            out, para = model(inputs)
            loss = loss_func(out, inputs)

            train_loss += loss.item() * inputs.size(0)
            loss.backward()

            metrics = {"train_batch_loss": loss}

            optimizer.step()
            if scheduler:
                scheduler.step()

        avg_train_loss = train_loss/train_data_size
        print("Training Loss: {:.4f}".format(avg_train_loss))
