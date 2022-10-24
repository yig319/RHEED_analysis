import os, glob, h5py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore


def load_curve(h5_para_file, growth, spot, para_key, show=True):
    h5_para = h5py.File(h5_para_file, mode='r')
    sound = h5_para[growth][spot][para_key][200:-200]
    if show:
        show_sound(sound, sound_fitted=None, colored_data=None, peaks=None,
                   xlabel='Time (1/500 s)', ylabel='Intensity (a.u.)', title=growth)
    return sound

def get_partials(sound, peaks):

    # get all partial sound 
    xs, ys = [], []
    for i in range(1, len(peaks)):
        xs.append(list(np.linspace(0, peaks[i]-peaks[i-1]-1-5, peaks[i]-peaks[i-1]-5)))
        ys.append(list(sound[5+peaks[i-1]:peaks[i]]))
    return xs, ys
        
def detect_peaks(sound, camera_freq, laser_freq, step_size, prominence, show=True):
    dist = int(camera_freq/laser_freq*0.6)

    # %matplotlib notebook
    step = np.hstack((np.ones(step_size), -1*np.ones(step_size)))
    dary_step = np.convolve(sound, step, mode='valid')
    dary_step = np.abs(dary_step)

    filtered_sound = dary_step/step_size

    peaks, properties = signal.find_peaks(dary_step, prominence=prominence, distance=dist)
    peaks = peaks[peaks>dist]
    peaks = peaks[peaks<len(sound)-dist]
    
    if show:
        show_sound(sound, sound_fitted=None, colored_data=None, peaks=peaks,
                   xlabel='Time (1/500 s)', ylabel='Intensity(a.u.)',
                   title='Curve with labelled peaks')
    return peaks

    
def fit_exp(xs, ys, b_, peaks=[], ylim=None, show=False):
    def exp_func_up(x, b):
        return (1 - np.exp(-x/b))
    def exp_func_down(x, b):
        return np.exp(-x/b)
    
    b_list, sign_list = [], []
    labels, ys_fit = [], []
    I_start_list, I_end_list = [], []
    
    for i in range(len(xs)):
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        
        # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
        n_avg = len(ys[i])//100+3
        I_start = np.mean(ys[i][:n_avg])
        I_end= np.mean(ys[i][-n_avg:])
        ys[i] = (ys[i]-I_start)/(I_end-I_start) # I-Imin/Imax-Imin (I/I0)
        
        params, params_covariance = curve_fit(exp_func_up, x, ys[i], p0=b_, bounds=[0.01, 1]) # curve_fit
        b = params[0]
        y_fit = exp_func_up(x, b)
        sign_list.append('1-exp(-t/tau)')
        labels.append(f'index: {i+1};\nfitted: 1-exp(-t/{np.round(b, 2)});\nstart at: {peaks[i]}')
        
        ys[i] = list(ys[i])
        b_list.append(np.round(b, 4))
        I_start_list.append(np.round(I_start, 4))
        I_end_list.append(np.round(I_end, 4))
        ys_fit.append(y_fit)

    if show:
        show_plots(ys, xs, labels, ys_fit, ylim=ylim)
    return b_list, sign_list, I_start_list, I_end_list

def remove_outlier(array, boundary, n=3):
    for n_ in range(n):
        z = zscore(array, axis=0, ddof=0)
        for i in range(1, len(array)-1):
            if z[i] < -np.abs(boundary) or z[i] > np.abs(boundary):
                array[i] = (array[i-1]+array[i+1])/2
    return array


def analyze_curve(h5_para_file, growth_name, spot, para_key, camera_freq, laser_freq, 
                  prominence, step_size, fit_ylim, color_type, show=True):
    
    if show:
        print(h5_para_file)
        print(growth_name, spot, metric)

    # load data
    sound = load_curve(h5_para_file, growth_name, spot, para_key, show=show)
    peaks = detect_peaks(sound, camera_freq, laser_freq, step_size=step_size, 
                         prominence=prominence, show=show)
    xs, ys = get_partials(sound, peaks)
    peaks_ = peaks[:-1]
    
    # fit exp
    b_list, sign_list, I_start_list, I_end_list = fit_exp(xs, ys, b_=0.1, peaks=peaks_, ylim=fit_ylim, show=show)
    
    # generate classified curves 
    if color_type == 'tau':
        sounds_color = color_curve(sound, b_list, peaks)
        b_color = color_curve(b_list, b_list, None)
    if color_type == 'function_type':
        sounds_color = color_curve(sound, sign_list, peaks)
        b_color = color_curve(b_list, sign_list, None)
        
    if show:
        show_sound(b_list, None, None, None, xlabel='Laser ablation (count)', 
                   ylabel='Diffusion Time Constant (s)', title=growth_name+'-'+spot+'-'+para_key)
#         show_sound(Imin_list, Imax_list, None, None, xlabel='Laser ablation (count)', 
#                    ylabel='RHEED Intensity (a.u.)', title=growth_name+'-'+spot+'-'+para_key+'-difference')
        
        plt.plot(I_start_list, marker='.', markersize=1, zorder=0)
        plt.plot(I_end_list, marker='.', markersize=1, zorder=5)
        plt.legend(['I_start', 'I_end'])
        plt.xlabel('Laser ablation (count)')
        plt.ylabel('RHEED Intensity (a.u.)')
        plt.title(growth_name+'-'+spot+'-'+para_key)
        plt.show()
        
        if color_type != None:
            show_sound(sound, None, sounds_color, peaks, xlabel='Time (1/500 s)', ylabel='Intensity(a.u.)',
                       title=growth_name+'-'+spot+'-'+para_key+'-colored data')
            show_sound(b_list, None, b_color, None, xlabel='Time (1/500 s)', ylabel='Intensity(a.u.)',
                       title=growth_name+'-'+spot+'-'+para_key+'-colored data')

    return sound, b_list, peaks, I_start_list, I_end_list

            
def color_curve(data, indicator, peaks=None):
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


def show_plots(ys, xs=None, labels=None, ys_fit=None, img_per_row=4, subplot_height=3, ylim=None):
    if type(labels) == type(None): labels = range(len(ys))
    if type(xs) == type(None):
        xs = []
        for y in ys:
            xs.append(np.linspace(0, len(y), len(y)+1))            
        
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


def label_violinplot(ax, df, xaxis, yaxis, xaxis_order, df_all=None, label_type='number'):
    # Calculate number of obs per group & median to position labels
    xloc = range(len(xaxis_order))
    yloc, text = [], [] 
    for i, element in enumerate(xaxis_order):
        yloc.append(df[df[xaxis]==element].tau.median())
        
        text.append('')
        if label_type == 'number':
            text[i] = "n: "+str(len(df[df[xaxis]==element]))
        
        if label_type == 'ratio':
            if dy_all == type(None):
                print('df_all is empty')
                return
            else:
                ratio = int(round(len(df[df[xaxis]==element]) / len(df_all[df_all[xaxis]==element]), 2)*100)
                text[i] = 'Ratio:\n'+str(len(df[df[xaxis]==element]))+'/'+str(len(df_all[df_all[xaxis]==element]))+'\n='+str(ratio)+'%'
        
        if label_type == 'mean':
            text[i] = str(round(df[df[xaxis]==element].tau.mean(), 4))

    if label_type in ['number', 'ratio']:
        for tick, label in zip(xloc, ax.get_xticklabels()):
            ax.text(xloc[tick], yloc[tick] + 0.03, text[tick], horizontalalignment='center', size=14, weight='semibold')
    
    if label_type in ['mean']:
        for tick, label in zip(xloc, ax.get_xticklabels()):
            ax.text(xloc[tick] + 0.03, yloc[tick] - 0.03, text[tick], horizontalalignment='left', size=14, weight='semibold')