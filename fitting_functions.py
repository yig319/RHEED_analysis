import h5py
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore

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

# def remove_outlier(array, boundary, n=3):
#     for n_ in range(n):
#         z = zscore(array, axis=0, ddof=0)
#         for i in range(1, len(array)-1):
#             if z[i] < -np.abs(boundary) or z[i] > np.abs(boundary):
#                 array[i] = (array[i-1]+array[i+1])/2
#     return array


def remove_outlier(x, y, boundary):
    z = zscore(y, axis=0, ddof=0)
    x = np.delete(x, np.where(z>boundary))
    y = np.delete(y, np.where(z>boundary))
    return x, y

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
      
def fit_exp(xs, ys, p_init, x_peaks=[], ylim=None, camera_freq=500, add_previous=False):
    def exp_func(x, tau):
        return (1 - np.exp(-x/tau))
    
    tau_list, sign_list = [], []
    labels, ys_fit = [], []
    I_start_list, I_end_list = [], []
    
    for i in range(len(xs)):
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        
        # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
        n_avg = len(ys[i])//100+3
        
        # add previous data point
        if i != 0 and add_previous:
            ys[i][0] = I_end
            
        I_start = np.mean(ys[i][:n_avg])
        I_end= np.mean(ys[i][-n_avg:])
            
        if I_end-I_start == 0: 
#             print('I_end = I_start, intensity:', ys[i])
            ys[i] = (ys[i]-I_start) # I-Imin/Imax-Imin (I/I0)
        else:
            ys[i] = (ys[i]-I_start)/(I_end-I_start) # I-Imin/Imax-Imin (I/I0)
            
        params, params_covariance = curve_fit(exp_func, x, ys[i], p0=p_init, bounds=[0.01, 1]) # curve_fit
        tau = params[0]
        y_fit = exp_func(x, tau)
        sign_list.append('1-exp(-t/tau)')
        labels.append(f'index: {i+1};\nfitted: 1-exp(-t/{np.round(tau, 2)})')
        
        ys[i] = list(ys[i])
        tau_list.append(np.round(tau, 4))
        I_start_list.append(np.round(I_start, 4))
        I_end_list.append(np.round(I_end, 4))
        ys_fit.append(y_fit)

    return tau_list, labels, ys_fit, [sign_list, I_start_list, I_end_list]



def fit_exp_function(xs, ys, camera_freq=500, 
                     fit_settings = {'fit_func': 'exp', 'I_diff': 5000, 'bounds':[0.01, 1], 'p_init':(1, 0.1),
                                     'from_previous': False, 'weight_previous': False, 'relative_intensity': False,
                                     'use_prev_as_bound': False}):
    
    '''
    I_diff: Intensity difference used to normalize the curve to 0-1;
    from_previous: int or False, the number of points used to add the I_end value in front of recovery curve;
    weight_previous: int or False, weight for previous data points: n times of average weight;
    
    '''
    def exp_func(x, a, b):
        return a*(1 - np.exp(-x/b))
    def linear_func(x, a, b):
        return a*x+b    
    
    fit_func = fit_settings['fit_func']
    I_diff = fit_settings['I_diff']
    bounds = fit_settings['bounds']
    p_init = fit_settings['p_init']
    from_previous = fit_settings['from_previous']
    weight_previous = fit_settings['weight_previous']
    relative_intensity = fit_settings['relative_intensity']
    use_prev_as_bound = fit_settings['use_prev_as_bound']
    
    if not from_previous and weight_previous:
        print("Please assign an integer to from_previous, otherwise, can't assign weight for previous data point.")
        return 
    
    if fit_func == 'linear' and weight_previous:
        print("Please note that the linear function will only be fitted with last 90% of data points of every curve, \
so the first several data points will not contribute to the fitting process.")
        
    a_list, b_list = [], []
    ys_nor, ys_nor_fit, ys_fit = [], [], []
    labels, I_drop_list = [], []
    
    for i in range(len(xs)):
        
        if i == 0:
            I_drop = 0
        else:
            I_drop = (I_end-np.mean(ys[i][:len(ys[i])//200+3]))/I_end
        
        # section: normalize the curve
        # add previous data point
        if i != 0 and from_previous:
            ys[i] = [I_end]*from_previous+ys[i]
            xs[i] = xs[i][:from_previous]+xs[i]

#             ys[i][:from_previous] = [I_end]*from_previous


        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        n_avg = len(ys[i])//100+3
        
        
        I_end = np.mean(ys[i][-n_avg:])
        if from_previous:
            I_start = np.mean(ys[i][:from_previous])
            if use_prev_as_bound: # choose smallest one as zero point so we can ignore out of range data points
                if I_end > I_start:
#                     print('upper:', [np.mean(ys[i][:n_avg]), I_start])
                    I_start = np.min([np.mean(ys[i][:n_avg]), I_start])
#                     print('I_start:', I_start)
                else:
#                     print('downward:', [np.mean(ys[i][:n_avg]), I_start])
                    I_start = np.max([np.mean(ys[i][:n_avg]), I_start]) 
#                     print('I_start:', I_start)
        else:
            I_start = np.mean(ys[i][:n_avg])
    

        # change the points from previous to be with relative intensity
        if relative_intensity:
            if from_previous and ((I_start < I_end and ys[i][0] > I_start) or (I_start > I_end and ys[i][0] < I_start)):
                if I_diff:
                    I_delta = np.abs((ys[i][0]-I_start)) * I_diff/np.abs(I_end-I_start)
                else:
                    I_delta = np.abs((ys[i][0]-I_start))
                ys[i][:from_previous] = [I_start - I_delta]*from_previous
            I_start = np.mean(ys[i][:n_avg])
        
        
        # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
        if I_end - I_start == 0: # avoid devide by 0
            y_nor = (ys[i]-I_start)
        elif I_diff: # use fixed I_diff
            if I_end < I_start:
                y_nor = (ys[i]-I_start)/(-I_diff)
            else:
                y_nor = (ys[i]-I_start)/I_diff # I-Imin/Imax-Imin (I/I0)

                
        # section: fit curve with function and collect data
        if fit_func == 'linear':
            params, params_covariance = curve_fit(linear_func, x[-int(len(x)*0.9):], y_nor[-int(len(x)*0.9):], p0=p_init) 
            a, b = params
            y_nor_fit = linear_func(x, a, b)
        
        elif fit_func == 'exp':
            # change weights for previous point
            if from_previous and weight_previous:
#                 yerr = 1/y_nor
#                 yerr[:from_previous]*=weight_previous
                yerr = np.ones(len(y_nor))*0.1
                yerr[int(len(y_nor)/5):]*=weight_previous
                print(yerr)
                
                params, params_covariance = curve_fit(exp_func, x, y_nor, p0=p_init, sigma=yerr, bounds=bounds, absolute_sigma=False) 
            else:
                params, params_covariance = curve_fit(exp_func, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a, b = params
            y_nor_fit = exp_func(x, a, b)

            if use_prev_as_bound and np.mean(y_nor[:from_previous]) > 0:
                if np.all(y_nor_fit < np.mean(y_nor[:from_previous])):
                    print('Empty range for use_prev_as_bound=True!!!')
                    print(y_nor_fit)
                    print('Lower limit:', np.mean(y_nor[:from_previous]))
                    return
                x_bound = np.min(np.where(y_nor_fit > np.mean(y_nor[:from_previous])))
                x = x[x_bound:]
            
                xs[i] = xs[i][x_bound:]
                ys[i] = ys[i][x_bound:]

                n_avg = len(ys[i])//100+3
                I_start = np.mean(ys[i][:n_avg])
                I_end = np.mean(ys[i][-n_avg:])
                
                # redo the normalization
                # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
                if I_end - I_start == 0:  # avoid devide by 0
                    y_nor = (ys[i]-I_start)
                elif I_diff:  # use fixed I_diff
                    if I_end < I_start:
                        y_nor = (ys[i]-I_start)/(-I_diff)
                    else:
                        y_nor = (ys[i]-I_start)/I_diff  # I-Imin/Imax-Imin (I/I0)
                
                # change weights for previous point
                if from_previous and weight_previous:
                    yerr = np.ones(len(y_nor))*0.1
                    yerr[int(len(y_nor)/5):]*=weight_previous
                    params, params_covariance = curve_fit(exp_func, x, y_nor, p0=p_init, sigma=yerr, bounds=bounds, absolute_sigma=False) 
                else:
                    params, params_covariance = curve_fit(exp_func, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
                a, b = params
                y_nor_fit = exp_func(x, a, b)

            # reverse calculation for y_fit
            if I_end - I_start == 0: # avoid devide by 0
                y_fit = y_nor_fit+I_start
            elif I_diff: # use fixed I_diff
                if I_end < I_start:
                    y_fit = y_nor_fit*(-I_diff)+I_start
                else:
                    y_fit = y_nor_fit*I_diff+I_start

        a_list.append(np.round(a, 4))
        b_list.append(np.round(b, 4))
        ys_fit.append(y_fit)
        ys_nor.append(y_nor)
        ys_nor_fit.append(y_nor_fit)
        labels.append(f'{np.round(a, 2)}*(1-exp(-t/{np.round(b, 2)}), I_drop: {np.round(I_drop, 2)}')
        I_drop_list.append(I_drop)

    return a_list, b_list, [xs, ys, ys_fit, ys_nor, ys_nor_fit, labels, I_drop_list]



def analyze_curves(h5_para_file, growth_dict, spot, metric, camera_freq=500, interval=1000, visualize=False,
                   fit_settings={'fit_func': 'exp', 'I_diff': 8000, 'bounds':[0.01, 1], 'p_init':(1, 0.1),
                                 'from_previous': 2, 'weight_previous': 1, 'relative_intensity': False}):
    '''
    h5_para_file: String, path of the h5 file name, containing information about the RHEED spots;
    growth_dict: names of the growth index and corresponding frequency;
    spot: String, name of the RHEED spot to collect, choice of "spot_1", "spot_2" or "spot_3"; 
    metric: String, name of the metric to analyze the RHEED spot, choice of ...;
    camera_freq: int, camera frequency of data acquisition. Default camera_freq is 500 fps;
    interval=1000;
    title=None;
    upper_limit: int, the number of levels to smooth the curve by remove the outliers. Default is 0;
    visualize=False;
    fit_settings: setting parameters for fitting function
    '''

    a_list_all, b_list_all, x_list_all = [], [], []
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all, I_drop_list_all = [], [], [], [], [], [], []
    
    x_end = 0
    for growth_name in list(growth_dict.keys()):

        # load data
        sample_x, sample_y = load_curve(h5_para_file, growth_name, 'spot_2', 'img_intensity', camera_freq=500, x_start=0)

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=growth_dict[growth_name],
                                       step_size=5, prominence=0.1)
        # fit exponential function
        a_list, b_list, info = fit_exp_function(xs, ys, camera_freq=500, fit_settings=fit_settings)        
        b_list_all+=b_list
        a_list_all+=a_list
        
        xs, ys, ys_fit, ys_nor, ys_nor_fit, labels, I_drop_list = info
        xs_all+=xs
        ys_all+=ys
        ys_fit_all+=ys_fit
        ys_nor_all+=ys_nor
        ys_nor_fit_all+=ys_nor_fit
        labels_all += labels
        I_drop_list_all += I_drop_list

        if x_list_all != []:
            x_list = x_peaks[:-1] + x_end
        else:
            x_list = x_peaks[:-1]

        if visualize:
            if x_list_all != []:
                sample_x += x_end
                x_peaks += x_end

            labels_dict = {}
            for i, x in enumerate(x_peaks[:-1]):
                labels_dict[x] = labels[i]

            plot_curve(np.concatenate(xs), np.concatenate(ys), curve_y_fit=np.concatenate(ys_fit), labels_dict=labels_dict,
                       plot_type='scatter', xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12, 4))  
            plot_curve(x_list, b_list, plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='Characteristic Time (s)', 
                       yaxis_style='linear', figsize=(12, 4))
            plot_curve(x_list, a_list, plot_type='lineplot', xlabel='Laser ablation (count)', ylabel='Intensity Magnitude (a.u.)', 
                       yaxis_style='linear', figsize=(12, 4))
            show_grid_plots(xs, ys_nor, labels, ys_nor_fit, ylim=None)
            show_grid_plots(xs, ys, labels, ys_fit, ylim=None)
        
        x_end = round(x_end + (len(sample_x)+interval)/camera_freq, 2)
        x_list_all.append(x_list)
        
    x_list_all = np.concatenate(x_list_all)[:len(b_list_all)]
    b_list_all = np.array(b_list_all)
    return a_list_all, b_list_all, x_list_all, [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all, I_drop_list_all]






def normalize_0_1(y, I_start, I_end, I_diff):
    # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
    if I_end - I_start == 0: # avoid devide by 0
        y_nor = (y-I_start)
    elif I_diff: # use fixed I_diff
        if I_end < I_start:
            y_nor = (y-I_start)/(-I_diff)
        else:
            y_nor = (y-I_start)/I_diff # I-Imin/Imax-Imin (I/I0)
    return y_nor

def de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff):
    # reverse calculation for y_fit
    if I_end - I_start == 0: # avoid devide by 0
        y_fit = y_nor_fit+I_start
    elif I_diff: # use fixed I_diff
        if I_end < I_start:
            y_fit = y_nor_fit*(-I_diff)+I_start
        else:
            y_fit = y_nor_fit*I_diff+I_start
    return y_fit


def process_rheed_data(xs, ys, I_diff, length=500, savgol_window_order=(15, 3), pca_component=10):
#     print(len(xs[0]), len(ys[0]))
    
    # normalize data
    ys_nor = []
    for y in ys:
        x = np.linspace(1e-5, 1, len(y)) # use second as x axis unit
        n_avg = len(y)//100+3
        I_end = np.mean(y[-n_avg:])
        I_start = np.mean(y[:n_avg])
        y_nor = normalize_0_1(y, I_start, I_end, I_diff)
        ys_nor.append(y_nor)
    
#     print('1:', len(xs[0]), len(ys_nor[0]))
    # interpolate the data to same size 
    if length:
        xs_sl = []
        ys_nor_sl = []
        for x, y_nor in zip(xs, ys_nor):
            x_sl = np.linspace(np.min(x), np.max(x), length)
#             print(len(x_sl), len(x), len(y_nor))
            y_sl = np.interp(x_sl, x, y_nor)
            xs_sl.append(x_sl)
            ys_nor_sl.append(y_sl)
    xs_sl, ys_nor_sl = np.array(xs_sl), np.array(ys_nor_sl)

    # denoise
    if savgol_window_order:
        ys_nor_sl_denoised = savgol_filter(ys_nor_sl, savgol_window_order[0], savgol_window_order[1])
    if pca_component:
        pca = PCA(n_components=pca_component)
        ys_nor_sl_denoised = pca.inverse_transform(pca.fit_transform(ys_nor_sl_denoised))
    return xs_sl, ys_nor_sl_denoised

