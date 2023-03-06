import h5py
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import torch

import sys
sys.path.append('./')
from visualize_functions import plot_curve, show_grid_plots

def load_curve(h5_para_file, growth, spot, metric, camera_freq, x_start):
    h5_para = h5py.File(h5_para_file, mode='r')
    curve_y = np.array(h5_para[growth][spot][metric])
    curve_x = np.linspace(x_start, x_start+len(curve_y)-1, len(curve_y))/camera_freq
    return curve_x, curve_y

def load_multiple_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, x_start=0, interval=200):
    x_all, y_all = [], []
    for growth_name in list(growth_dict.keys()):
        curve_x, curve_y = load_curve(h5_para_file, growth_name, spot, metric, camera_freq, x_start)
        x_start = x_start+len(curve_y)+interval
        x_all.append(curve_x)
        y_all.append(curve_y)
        
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    return x_all, y_all


def detect_peaks(curve_x, curve_y, camera_freq, laser_freq, step_size, prominence):
    dist = int(camera_freq/laser_freq*0.6)
    step = np.hstack((np.ones(step_size), -1*np.ones(step_size)))
    dary_step = np.convolve(curve_y, step, mode='valid')
    dary_step = np.abs(dary_step)

    filtered_curve_y = dary_step/step_size
    x_peaks, properties = signal.find_peaks(dary_step, prominence=prominence, distance=dist)
    x_peaks = x_peaks[x_peaks>dist]
    x_peaks = x_peaks[x_peaks<len(curve_y)-dist]
    
    # get all partial curve 
    xs, ys = [], []
    for i in range(1, len(x_peaks)):
        xs.append(list(curve_x[5+x_peaks[i-1]:x_peaks[i]]))
        ys.append(list(curve_y[5+x_peaks[i-1]:x_peaks[i]]))
    return x_peaks/500, xs, ys

def remove_outlier(x, y, ub):
    z = zscore(y, axis=0, ddof=0)
    x = np.delete(x, np.where(z>ub))
    y = np.delete(y, np.where(z>ub))
    return x, y

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def normalize_0_1(y, I_start, I_end, I_diff=None, unify=True):
    if not I_diff:
        I_diff = I_end-I_start
    
    # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
    if I_end - I_start == 0: # avoid devide by 0
        y_nor = (y-I_start)
    elif unify:
        if I_end < I_start:
            y_nor = (I_start-y)/I_diff
        else:
            y_nor = (y-I_start)/I_diff
    else:
        if I_end < I_start:
            y_nor = (y-I_end)/I_diff
        else:
            y_nor = (y-I_start)/I_diff
    return y_nor

def de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff=None, unify=True):
    
    if not I_diff:
        I_diff = I_end-I_start
    if not unify:
        I_diff = np.abs(I_diff)
    
    # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
    if I_end - I_start == 0: # avoid devide by 0
        y_nor = (y-I_start)
    elif unify:
        if I_end < I_start:
            y_fit = I_start-y_nor_fit*I_diff
        else:
            y_fit = y_nor_fit*I_diff+I_start

    else:
        if I_end < I_start:
            y_fit = y_nor_fit*I_diff+I_end
        else:
            y_fit = y_nor_fit*I_diff+I_start
    return y_fit
    

def process_rheed_data(xs, ys, length=500, savgol_window_order=(15, 3), pca_component=10):    
    # interpolate the data to same size 
    if length:
        xs_processed = []
        ys_processed = []
        for x, y in zip(xs, ys):
            x_sl = np.linspace(np.min(x), np.max(x), length)
            y_sl = np.interp(x_sl, x, y)
            xs_processed.append(x_sl)
            ys_processed.append(y_sl)
    xs_processed, ys_processed = np.array(xs_processed), np.array(ys_processed)

    # denoise
    if savgol_window_order:
        ys_processed = savgol_filter(ys_processed, savgol_window_order[0], savgol_window_order[1])
    if pca_component:
        pca = PCA(n_components=pca_component)
        ys_processed = pca.inverse_transform(pca.fit_transform(ys_processed))
    return xs_processed, ys_processed


def fit_exp_function(xs, ys, fit_settings = {'I_diff': 5000, 'unify': True, 'bounds':[0.01, 1], 'p_init':(1, 0.1)}):
    '''
    I_diff: Intensity difference used to normalize the curve to 0-1;
    
    '''
    def exp_func_inc(x, a1, b1, relax1):
        return (a1*x+b1)*(1 - np.exp(-x/relax1))
    
    def exp_func_dec(x, a2, b2, relax2):
        return (a2*x+b2)*np.exp(-x/relax2)
  

    I_diff = fit_settings['I_diff']
    bounds = fit_settings['bounds']
    p_init = fit_settings['p_init']
    unify = fit_settings['unify']

    parameters = []
    ys_nor, ys_nor_fit, ys_fit = [], [], []
    labels, losses = [], []
    
    for i in range(len(xs)):
        
        # section: normalize the curve
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        n_avg = len(ys[i])//100+3
        I_end = np.mean(ys[i][-n_avg:])
        I_start = np.mean(ys[i][:n_avg])
        y_nor = normalize_0_1(ys[i], I_start, I_end, I_diff, unify)
        
        if unify:
            params, params_covariance = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a, b, relax = params
            y_nor_fit = exp_func_inc(x, a, b, relax)
            labels.append(f'index {i+1}:\ny=({np.round(a, 2)}t+{np.round(b, 2)})*(1-exp(-t/{np.round(relax, 2)}))')
            parameters.append((a, b, relax))
            losses.append((0, 0))

            
        else:
            params, params_covariance = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a1, b1, relax1 = params
            y1_nor_fit = exp_func_inc(x, a1, b1, relax1)

            params, params_covariance = curve_fit(exp_func_dec, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a2, b2, relax2 = params
            y2_nor_fit = exp_func_dec(x, a2, b2, relax2)

            loss1 = ((y_nor - y1_nor_fit)**2).mean()
            loss2 = ((y_nor - y2_nor_fit)**2).mean()

            if loss1 < loss2:
                y_nor_fit = y1_nor_fit
                labels.append(f'index {i+1}:\ny1=({np.round(a1, 2)}t+{np.round(b1, 2)})*(1-exp(-t/{np.round(relax1, 2)}))')
                parameters.append((a1, b1, relax1))
                
            else:
                y_nor_fit = y2_nor_fit
                labels.append(f'index {i+1}:\ny2=({np.round(a2, 2)}t+{np.round(b2, 2)})*(exp(-t/{np.round(relax2, 2)}))')
                parameters.append((a2, b2, relax2))
                
            losses.append((loss1, loss2))

#         y_nor_fit = y1_nor_fit
        y_fit = de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff, unify)
        
        ys_fit.append(y_fit)
        ys_nor.append(y_nor)
        ys_nor_fit.append(y_nor_fit)
    return np.array(parameters), [xs, ys, ys_fit, ys_nor, ys_nor_fit, labels, losses]


def analyze_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, interval=1000, visualize=False, fit_settings={'savgol_window_order': (15,3), 'pca_component': 10, 'I_diff': 8000, 'unify':True, 'bounds':[0.01, 1], 'p_init':(1, 0.1)}):

    '''
    Analyzes RHEED curves for a given spot and metric.

    Args:
    - h5_para_file (str): Path of the h5 file name, containing information about the RHEED spots.
    - growth_dict (dict): Names of the growth index and corresponding frequency.
    - spot (str): Name of the RHEED spot to collect, choice of "spot_1", "spot_2" or "spot_3".
    - metric (str): Name of the metric to analyze the RHEED spot.
    - camera_freq (int): Camera frequency of data acquisition. Default camera_freq is 500 fps.
    - interval (int): Interval between two consecutive RHEED curves.
    - visualize (bool): If True, plots the analyzed data. Default is False.
    - fit_settings (dict): Setting parameters for fitting function.

    Returns:
    - parameters_all (ndarray): Fitted parameters for all RHEED curves.
    - x_list_all (ndarray): Laser ablation counts for all RHEED curves.
    - info_all (list): List containing all processed RHEED data.
    '''

    parameters_all, x_list_all = [], []
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all, losses_all = [], [], [], [], [], [], []
    
    x_end = 0
    for growth_name in list(growth_dict.keys()):

        # load data
        sample_x, sample_y = load_curve(h5_para_file, growth_name, 'spot_2', 'img_intensity', camera_freq=500, x_start=0)

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=growth_dict[growth_name],
                                       step_size=5, prominence=0.1)
        
        xs, ys = process_rheed_data(xs, ys, length=500, savgol_window_order=fit_settings['savgol_window_order'], pca_component=fit_settings['pca_component'])        

        
        # fit exponential function
        parameters, info = fit_exp_function(xs, ys, fit_settings=fit_settings)        
        parameters_all.append(parameters)
        xs, ys, ys_fit, ys_nor, ys_nor_fit, labels, losses = info
        xs_all.append(xs)
        ys_all.append(ys)
#         xs_all+=xs
#         ys_all+=ys
        ys_fit_all+=ys_fit
        ys_nor_all+=ys_nor
        ys_nor_fit_all+=ys_nor_fit
        labels_all += labels
        losses_all += losses

#         x_list = np.copy(x_peaks[:-1])
#         if x_list_all == []: # first growth 
#             x_list = x_peaks[:-1]
#         else:
        x_list = x_peaks[:-1] + x_end
        x_end = round(x_end + (len(sample_x)+interval)/camera_freq, 2)
        x_list_all.append(x_list)
        
        if visualize:
            print(growth_name, ':')

            labels_dict = {}
            for i, x in enumerate(x_peaks[:-1]):
                labels_dict[x] = labels[i]
            plot_curve(np.concatenate(xs), np.concatenate(ys), curve_y_fit=np.concatenate(ys_fit), labels_dict=labels_dict,
                       plot_type='scatter', xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12, 4))  
            if fit_settings['unify'] == False:
                losses_np = np.array(losses)
                
                x_y1 = x_peaks[:-1][losses_np[:,0]>losses_np[:,1]]
                loss_y1 = losses_np[:,0][losses_np[:,0]>losses_np[:,1]]
                x_y2 = x_peaks[:-1][losses_np[:,0]<losses_np[:,1]]
                loss_y2 = losses_np[:,1][losses_np[:,0]<losses_np[:,1]]
                plot_curve(x_y1, loss_y1, curve_x_fit=x_y2, curve_y_fit=loss_y2, plot_type='scatter', ylim=(-0.005, 0.01), xlabel='Laser ablation (count)', ylabel='MSE Loss (a.u.)', yaxis_style='linear', 
                           legend=['y1=(ax+b)*(1-np.exp(-x/tau))', 'y2=(ax+b)*np.exp(-x/tau)'], figsize=(12, 4), title='Indicator for fitting function choice')


            plot_curve(x_peaks[:-1], parameters[:,0], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='Intensity Magnitude Constant (a.u.)', yaxis_style='linear', figsize=(12, 4))
            plot_curve(x_peaks[:-1], parameters[:,1], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='Intensity Magnitude Linear (a.u.)', yaxis_style='linear', figsize=(12, 4))
            plot_curve(x_peaks[:-1], parameters[:,2], plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='Characteristic Time (s)', yaxis_style='linear', figsize=(12, 4))
            
            print('Fitting curves for normalized data:')
            show_grid_plots(xs, ys_nor, labels, ys_nor_fit, ylim=None)
            print('Fitting curves for raw data:')
            show_grid_plots(xs, ys, labels, ys_fit, ylim=None)
        

    parameters_all = np.concatenate(parameters_all, 0)
    x_list_all = np.concatenate(x_list_all)[:len(parameters_all)]
    return parameters_all, x_list_all, [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all, np.array(losses_all)]