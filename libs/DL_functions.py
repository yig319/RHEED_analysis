# model and training functions

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import wandb
import sys
sys.path.append('./')
from visualize_functions import plot_curve, show_grid_plots
from fitting_functions import load_curve, detect_peaks, process_rheed_data, normalize_0_1, de_normalize_0_1



def prepare_dataloader(h5_para_file, spot, metric, growth_dict, camera_freq, length, I_diff, unify, batch_size, savgol_window_order=(15,3), pca_component=10, visualize=False):
    xs_processed_all, ys_processed_all, xs_nor_all, ys_nor_all = [], [], [], []
    for growth_name in list(growth_dict.keys()):
        # load data
        sample_x, sample_y = load_curve(h5_para_file, growth_name, spot, metric, camera_freq=camera_freq, x_start=0)

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=growth_dict[growth_name], step_size=5, prominence=0.1)
        
        xs_processed, ys_processed = process_rheed_data(xs, ys, length=length, savgol_window_order=(15,3), pca_component=10)        

        ys_nor = []
        for x, y in zip(xs_processed, ys_processed):
            n_avg = len(y)//100+3
            I_end = np.mean(y[-n_avg:])
            I_start = np.mean(y[:n_avg])
            y_nor = normalize_0_1(y, I_start, I_end, I_diff, unify)
            ys_nor.append(y_nor)
        xs_processed_all.append(xs_processed)
        ys_processed_all.append(ys_processed)
        ys_nor_all.append(ys_nor)
        
    xs_processed = np.concatenate(xs_processed, 0)
    ys_processed = np.concatenate(ys_processed, 0)
    ys_nor_all = np.concatenate(ys_nor_all, 0)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(ys_nor_all).unsqueeze(1))
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dl, xs_processed, ys_nor_all


class linear_fit_model(nn.Module):
    def __init__(self):
        super().__init__()  
        self.nn = nn.Sequential(nn.BatchNorm1d(1),
                                nn.Linear(500, 1000),
                                nn.LeakyReLU(),
                                nn.BatchNorm1d(1),
                                nn.Linear(1000, 2000),
                                nn.LeakyReLU(),
                                nn.BatchNorm1d(1),
                                nn.Linear(2000, 2000),
                                nn.LeakyReLU(),
                                nn.BatchNorm1d(1),
                                nn.Linear(2000, 500),
                                nn.LeakyReLU(),
                                nn.Linear(500, 3),
                                nn.Sigmoid())
    def forward(self, x):
        parameters = self.nn(x).squeeze()
        m1, m2, relax = parameters[:, 0], parameters[:, 1], parameters[:, 2] # 02/23/2023-09:57

        t = torch.linspace(0, 1, 500).unsqueeze(1).repeat(1, len(x)).to(torch.float32).to(x.device)
        y_fit = (m1 + m2*t)*(1-torch.exp(-relax*10 * t))
        y_fit = torch.swapaxes(y_fit.unsqueeze(1), 0, 2)
        return y_fit, parameters
    
    
    
def train(model, train_dl, epochs, start, device, loss_func, optimizer, scheduler=None, save_path=None, tracking=False):
    model.train()
    train_data_size = len(train_dl.dataset)
    if tracking:   
        wandb.watch(model, log_freq=100)
        
    for epoch_idx in range(start, epochs+start):
        print("Epoch: {}/{}".format(epoch_idx+1, epochs+start))

        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_dl)):
            inputs = batch[0]
            inputs = inputs.to(torch.float32).to(device)
            model = model.to(device)

            optimizer.zero_grad()
            out, parameters = model(inputs)
            loss = loss_func(out, inputs)

#             topk = parameters[:, 0, -2:]
#             penalty = loss_func(torch.sum(topk, -1).squeeze(),  torch.ones(topk.shape[0]).to(device))
#             loss = loss_func(out, inputs) + penalty # 02/23/2023-09:57

            
            
            train_loss += loss.item() * inputs.size(0)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

        avg_train_loss = train_loss/train_data_size
        metrics = {"epoch": epoch_idx,
                    "train_loss": avg_train_loss}
        if tracking: wandb.log(metrics)

        print("Training Loss: {:.4f}".format(avg_train_loss))
        torch.save(model, save_path)
        
        
        
def dl_label(parameters_all):
    labels = []
    for i, parameters in enumerate(parameters_all):
        parameters_str = []
        for p in parameters:
            parameters_str.append(str(p))
        m1, m2, relax = parameters_str
        
        labels.append(f'index: {i+1}\ny=')
        if (m1!=str(0.0) and m2!=str(0.0)): 
            labels[i] += f'({m1}+{m2}*t)*(1-exp(t/{relax}))'
        elif m1!=str(0.0) and m2==str(0.0): 
            labels[i] += f'{m1}*(1-exp(t/{relax}))'
        elif m2!=str(0.0) and m1==str(0.0): 
            labels[i] += f'({m2}*t)*(1-exp(t/{relax}))'

        if labels[i][-1] == '+': labels[i] = labels[i][:-1]
        if labels[i][-1] == '=': labels[i] += 'constant'
    return labels
    

def predict_parameters(xs, ys, length=500, savgol_window_order=(15,3), pca_component=10, I_diff=12000, unify=True, model_path=None, device='cpu'):
    xs_processed, ys_processed = process_rheed_data(xs, ys, length=length, savgol_window_order=savgol_window_order, pca_component=pca_component)        
    
    # section: normalize the curve
    ys_nor = []
    for x, y in zip(xs_processed, ys_processed):
        n_avg = len(y)//100+3
        I_end = np.mean(y[-n_avg:])
        I_start = np.mean(y[:n_avg])
        
        y_nor = normalize_0_1(y, I_start, I_end, I_diff, unify)
        ys_nor.append(y_nor)
        
    model = torch.load(model_path).to(device)
    model = model.eval()
    inputs = torch.tensor(ys_nor).to(device).to(torch.float32).unsqueeze(1)
    ys_nor_fit, parameters = model(inputs)
#     ys_nor_fit, x_processed, parameters_np = model(inputs) # swap parameters and x_processed
#     print(x_processed, parameters_np)
    
    parameters_np = np.round(parameters.squeeze().detach().numpy(), 2)
    with np.errstate(divide='ignore'):
        parameters_np[:,-3] = np.round(1/parameters_np[:,-3]/10, 2)

#     print(parameters_np)
    labels = dl_label(parameters_np)
    ys_nor_fit = ys_nor_fit.detach().cpu().squeeze().numpy()
    
    # reverse calculation for y_fit
    ys_fit = []
    for x, y, y_nor_fit in zip(xs_processed, ys_processed, ys_nor_fit):
        n_avg = len(y)//100+3
        I_end = np.mean(y[-n_avg:])
        I_start = np.mean(y[:n_avg])
        y_fit = de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff, unify)
        ys_fit.append(y_fit)
    ys_fit = np.array(ys_fit)
    return parameters_np, [xs, ys, xs_processed, ys_processed, ys_fit, ys_nor, ys_nor_fit, labels]


def analyze_curves_dl(h5_para_file, growth_dict, spot, metric, camera_freq=500, interval=1000, visualize=False, length=500, savgol_window_order=(15,3), pca_component=10, I_diff=12000, unify=True, model_path=None):
    '''
    h5_para_file: String, path of the h5 file name, containing information about the RHEED spots;
    growth_dict: names of the growth index and corresponding frequency;
    spot: String, name of the RHEED spot to collect, choice of "spot_1", "spot_2" or "spot_3"; 
    metric: String, name of the metric to analyze the RHEED spot, choice of ...;
    camera_freq: int, camera frequency of data acquisition. Default camera_freq is 500 fps;
    interval=1000;
    visualize=False;
    '''

    parameters_all, x_coor_all = [], []
    xs_all, ys_all, xs_processed_all, ys_processed_all = [], [], [], []
    ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all = [], [], [], []
    
    x_end = 0
    for i, growth_name in enumerate(list(growth_dict.keys())):
        # load data
        sample_x, sample_y = load_curve(h5_para_file, growth_name, 'spot_2', 'img_intensity', camera_freq=500, x_start=0)

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=growth_dict[growth_name],
                                       step_size=5, prominence=0.1)
        
        # fit exponential function
        if type(model_path) == list:
            model_name = model_path[i]
        else:
            model_name = model_path
            
        parameters_np, info = predict_parameters(xs, ys, length=length, savgol_window_order=savgol_window_order, pca_component=pca_component, I_diff=I_diff, unify=unify, model_path=model_name, device='cpu')      
        
        parameters_all.append(parameters_np)
        xs, ys, xs_processed, ys_processed, ys_fit, ys_nor, ys_nor_fit, labels = info
        
        # get the x coordinates for fitting parameters 
        if x_coor_all != []: x_coor = np.array(x_peaks[:-1] + x_end)
        else: x_coor = np.array(x_peaks[:-1])
        if i>0: xs = [x+x_coor[0] for x in xs]

        xs_all+=xs
        ys_all+=ys
        xs_processed_all.append(xs_processed)
        ys_processed_all.append(ys_processed)
        ys_fit_all.append(ys_fit)
        ys_nor_all.append(ys_nor)
        ys_nor_fit_all.append(ys_nor_fit)
        labels_all += labels

        if visualize:
            print(growth_name, ':')
            labels_dict = {}
            for i, x in enumerate(x_peaks[:-1]): labels_dict[x] = labels[i]
                
            plot_curve(np.concatenate(xs_processed), np.concatenate(ys_processed), curve_y_fit=np.concatenate(ys_fit), labels_dict=labels_dict,
                       plot_type='scatter', xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12, 4))   
            print('MSE loss: ', nn.MSELoss()(torch.tensor(ys_nor).to(torch.float32), torch.tensor(ys_nor_fit).to(torch.float32)))
            
            plot_curve(x_coor, parameters_np[:, 0], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='', 
                       yaxis_style='sci', figsize=(12, 4))
            plot_curve(x_coor, parameters_np[:, 1], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='m2', 
                       yaxis_style='sci', figsize=(12, 4))
            plot_curve(x_coor, parameters_np[:, 2], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='relax', 
                       yaxis_style='sci', figsize=(12, 4))

            show_grid_plots(xs_processed, ys_nor, labels=labels, ys_fit=ys_nor_fit, subplot_height=3)
        
        x_end = round(x_end + (len(sample_x)+interval)/camera_freq, 2)
        x_coor_all.append(x_coor)
        
    parameters_all = np.concatenate(parameters_all, 0)
    x_coor_all = np.concatenate(x_coor_all)[:len(parameters_all)]
    return parameters_all, x_coor_all, \
            [xs_all, ys_all, xs_processed_all, ys_processed_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all]


def visualize_predictions(x_all, y_all, parameters_all, x_coor_all, info):
    [xs_all, ys_all, xs_processed_all, ys_processed_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all] = info
    
    x_coor_all = np.copy(x_coor_all)
    parameters = np.copy(parameters_all)

    plot_curve(x_all, y_all, xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12,2.5), xlim=(-2, 135))

    plot_curve(x_coor_all, parameters_all[:,0], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m1 (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_all[:,1], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m2*x (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_all[:,2], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: Characteristic Time (s)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))

    print('MSE loss for DL model fitting is:', nn.MSELoss()(torch.tensor(np.concatenate(ys_nor_all, 0)), 
                                                            torch.tensor(np.concatenate(ys_nor_fit_all, 0))).item())