import os, glob, h5py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore
import seaborn as sns

def NormalizeData(data, lb=0, ub=1):
    return (data - lb) / (ub - lb)


def trim_axes(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def show_images(images, labels=None, img_per_row=8, img_height=1, colorbar=False, 
                clim=False, scale_0_1=False, hist_bins=None, show_axis=False):
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"

    def scale(x):
        if x.min() < 0:
            return (x - x.min()) / (x.max() - x.min())
        else:
            return x/(x.max() - x.min())
    
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    n = 1
    if hist_bins: n +=1
        
    fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, n*h*len(images)//img_per_row+1))
    trim_axes(axes, len(images))

    for i, img in enumerate(images):
        
        if scale_0_1: img = scale(img)
        
        if len(images) <= img_per_row and not hist_bins:
            index = i%img_per_row
        else:
            index = (i//img_per_row)*n, i%img_per_row

        axes[index].title.set_text(labels[i])
        im = axes[index].imshow(img)
        if colorbar:
            fig.colorbar(im, ax=axes[index])
            
        if clim:
            m, s = np.mean(img), np.std(img)            
            im.set_clim(m-3*s, m+3*s) 
            
        if not show_axis:
            axes[index].axis('off')

        if hist_bins:
            index_hist = (i//img_per_row)*n+1, i%img_per_row
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)
    plt.show()
    
    
    
    
def plot_curve(curve_x, curve_y, x_peaks=None, plot_type='scatter', xlabel=None, ylabel=None, 
               xlim=None, ylim=None, yaxis_style='sci', title=None, figsize=(12,2.5), save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if plot_type == 'scatter':
        plt.scatter(x=curve_x, y=curve_y, c='k', s=1)
    if plot_type == 'lineplot':
        plt.plot(curve_x, curve_y, color='k', marker='.')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if type(xlim) != type(None): ax.set_xlim(xlim)
    if type(ylim) != type(None): ax.set_xlim(ylim)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if type(x_peaks) != type(None):
        for x in x_peaks:
            y = curve_y[np.where(curve_x==x)]
            pl.text(x, y, str(x), color="red", fontsize=6)
    if yaxis_style == 'sci':
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)
    plt.show()
    
    
def load_curve(h5_para_file, growth, spot, metric, camera_freq, x_start, figsize):
    h5_para = h5py.File(h5_para_file, mode='r')
    curve_y = h5_para[growth][spot][metric][200:-200]
    curve_x = np.linspace(x_start, x_start+len(curve_y)-1, len(curve_y))/camera_freq
    return curve_x, curve_y


def load_multiple_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, 
                         x_start=0, interval=200, title=None, figsize=(12,2.5), save_path=None):
    x_all, y_all = [], []
    for growth_name in list(growth_dict.keys()):
        curve_x, curve_y = load_curve(h5_para_file, growth_name, spot, metric, camera_freq, x_start, figsize=figsize)
        x_start = x_start+len(curve_y)+interval
        x_all.append(curve_x)
        y_all.append(curve_y)
        
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    return x_all, y_all
    
    
    
    
    
    
    
def color_curve(data, indicator, peaks=None):
    
    '''
    # example to generate classified curves 
    curve_classified_tau = color_curve(data=curve, indicator=tau_list, peaks=peaks)
    tau_classified_tau = color_curve(data=tau_list, indicator=tau_list, peaks=None)

    curve_classified_functype = color_curve(data=curve, indicator=sign_list, peaks=peaks)
    tau_classified_functype = color_curve(data=tau_list, indicator=sign_list, peaks=None)

    '''
    x = np.linspace(0, len(data)-1, len(data))
    x_up, y_up, x_down, y_down = [], [], [], []
    for i in range(len(indicator)):
        if (type(indicator[i])!=str and indicator[i] < 0.8) or (type(indicator[i])==str and indicator[i] == '1-exp(-t/tau)'):
            if type(peaks)!=type(None):
                x_up+=list(x[peaks[i]: peaks[i+1]])
                y_up+=list(data[peaks[i]: peaks[i+1]])
            else:
                x_up.append(x[i])
                y_up.append(data[i])
                
        if (type(indicator[i])!=str and indicator[i] >= 0.8) or (type(indicator[i])==str and indicator[i] == 'exp(-t/tau)'):
            if type(peaks)!=type(None):
                x_down+=list(x[peaks[i]: peaks[i+1]])
                y_down+=list(data[peaks[i]: peaks[i+1]])
            else:
                x_down.append(x[i])
                y_down.append(data[i])
            
    return x_up, y_up, x_down, y_down      


def fit_exp(xs, ys, tau_, x_peaks=[], ylim=None, camera_freq=500):
    def exp_func(x, tau):
        return (1 - np.exp(-x/tau))
    
    tau_list, sign_list = [], []
    labels, ys_fit = [], []
    I_start_list, I_end_list = [], []
    
    for i in range(len(xs)):
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        
        # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
        n_avg = len(ys[i])//100+3
        I_start = np.mean(ys[i][:n_avg])
        I_end= np.mean(ys[i][-n_avg:])
        
        
#         if I_end-I_start <= 0: continue
#         print(i)
        
        ys[i] = (ys[i]-I_start)/(I_end-I_start) # I-Imin/Imax-Imin (I/I0)
        params, params_covariance = curve_fit(exp_func, x, ys[i], p0=tau_, bounds=[0.01, 1]) # curve_fit
        tau = params[0]
        y_fit = exp_func(x, tau)
        sign_list.append('1-exp(-t/tau)')
        labels.append(f'index: {i+1};\nfitted: 1-exp(-t/{np.round(tau, 2)});\nstart at: {camera_freq*x_peaks[i]}')
        
        ys[i] = list(ys[i])
        tau_list.append(np.round(tau, 4))
        I_start_list.append(np.round(I_start, 4))
        I_end_list.append(np.round(I_end, 4))
        ys_fit.append(y_fit)

    return tau_list, labels, ys_fit, [sign_list, I_start_list, I_end_list]





def show_grid_plots(xs, ys, labels=None, ys_fit=None, img_per_row=4, subplot_height=3, ylim=None):

    if type(labels) == type(None): labels = range(len(ys))

    fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
                             figsize=(16, subplot_height*len(ys)//img_per_row+1))    
    trim_axes(axes, len(ys))
    
    for i in range(len(ys)):
        
        if len(ys) <= img_per_row:
            index = i%img_per_row
        else:
            index = (i//img_per_row), i%img_per_row

        axes[index].title.set_text(labels[i])
        
        im = axes[index].plot(xs[i], ys[i], marker='.')
        
        if type(ys_fit) != type(None):
            im = axes[index].plot(xs[i], ys_fit[i])
        
        if type(ylim) != type(None):
            axes[index].set_ylim([ylim[0], ylim[1]])

    fig.tight_layout()
    plt.show()

    
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


    
    
def remove_outlier(array, boundary, n=3):
    for n_ in range(n):
        z = zscore(array, axis=0, ddof=0)
        for i in range(1, len(array)-1):
            if z[i] < -np.abs(boundary) or z[i] > np.abs(boundary):
                array[i] = (array[i-1]+array[i+1])/2
    return array




def fit_exp(xs, ys, tau_, x_peaks=[], ylim=None, camera_freq=500):
    def exp_func(x, tau):
        return (1 - np.exp(-x/tau))
    
    tau_list, sign_list = [], []
    labels, ys_fit = [], []
    I_start_list, I_end_list = [], []
    
    for i in range(len(xs)):
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        
        # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
        n_avg = len(ys[i])//100+3
        I_start = np.mean(ys[i][:n_avg])
        I_end= np.mean(ys[i][-n_avg:])
        
        
        if I_end-I_start <= 0: continue
#         print(i)
        
        ys[i] = (ys[i]-I_start)/(I_end-I_start) # I-Imin/Imax-Imin (I/I0)
        params, params_covariance = curve_fit(exp_func, x, ys[i], p0=tau_, bounds=[0.01, 1]) # curve_fit
        tau = params[0]
        y_fit = exp_func(x, tau)
        sign_list.append('1-exp(-t/tau)')
        labels.append(f'index: {i+1};\nfitted: 1-exp(-t/{np.round(tau, 2)});\nstart at: {camera_freq*x_peaks[i]}')
        
        ys[i] = list(ys[i])
        tau_list.append(np.round(tau, 4))
        I_start_list.append(np.round(I_start, 4))
        I_end_list.append(np.round(I_end, 4))
        ys_fit.append(y_fit)

    return tau_list, labels, ys_fit, [sign_list, I_start_list, I_end_list]



def analyze_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, title=None):
    
    tau_list_all, Imin_list_all, Imax_list_all = [], [], []
    for growth_name in list(growth_dict.keys()):
        # load data
        sample_x, sample_y = load_curve(h5_para_file, growth_name, 'spot_2', 'img_intensity', camera_freq=500, x_start=0, figsize=(12,4))

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=growth_dict[growth_name],
                                       step_size=5, prominence=0.1)
#         print(np.sum(x_peaks))

        # fit exponential function
        tau_list, labels, ys_fit, info = fit_exp(xs, ys, tau_=0.1, x_peaks=x_peaks, ylim=None, camera_freq=camera_freq)
        sign_list, I_start_list, I_end_list = info
        tau_list_all+=tau_list

    tau_list_all = remove_outlier(np.array(tau_list_all[2:]), 0.8, n=5)

    return tau_list_all