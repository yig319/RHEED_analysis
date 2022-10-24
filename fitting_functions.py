import h5py
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore


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
        
        if I_end-I_start == 0: 
#             print('I_end = I_start, intensity:', ys[i])
            ys[i] = (ys[i]-I_start) # I-Imin/Imax-Imin (I/I0)
        else:
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