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