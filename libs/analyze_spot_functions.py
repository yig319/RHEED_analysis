import os, h5py, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, sosfilt, freqz
from scipy import optimize
from joblib import Parallel, delayed

sys.path.append('./')
from visualize_functions import show_images


def NormalizeData(data, lb=0, ub=1):
    return (data - lb) / (ub - lb)
    
def show_fft_frequency(amplitude, samplingFrequency, ranges=None):

    # Frequency domain representation
    fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency

    tpCount = len(amplitude)
    values = np.arange(int(tpCount/2))
    timePeriod = tpCount/samplingFrequency

    frequencies = values/timePeriod
    fourierTransform[abs(fourierTransform)>1] = 0 
    if ranges:
        frequencies_ = frequencies[frequencies>ranges[0]]
        fourierTransform_ = fourierTransform[frequencies>ranges[0]]

        frequencies_range = frequencies_[frequencies_<ranges[1]-ranges[0]]
        fourierTransform_range = fourierTransform_[frequencies_<ranges[1]-ranges[0]]
    else:
        frequencies_range = frequencies
        fourierTransform_range = fourierTransform
        
    plt.figure(figsize=(15,4))
    plt.plot(frequencies_range, abs(fourierTransform_range))
    plt.show()
    return frequencies_range, abs(fourierTransform_range)

def butter_filter(data, method, filter_type, cutoff, samplingFrequency, order):
    nyq = 0.5 * samplingFrequency
    
    if type(cutoff)==tuple:
        cutoff = list(cutoff)
        
    if type(cutoff)==list:
        cutoff[0] = cutoff[0] / nyq
        cutoff[1] = cutoff[1] / nyq
    else:
        cutoff = cutoff / nyq

    if method == 'ba':
        b, a = butter(order, cutoff, btype=filter_type, analog=False, output='ba')
        y = lfilter(b, a, data)
    if method == 'sos':
        sos = butter(order, cutoff, btype=filter_type, analog=False, output='sos')
        y = sosfilt(sos, data)
    return y

def process_pass_filter(sound, filter_type, method, cutoff, order, frame_range, samplingFrequency=100):
    sig = np.copy(sound)
    t = np.arange(0, len(sig))
    ranges = None

    filtered = butter_filter(sig, method, filter_type, cutoff, samplingFrequency, order)
        
    show_fft_frequency(sig, samplingFrequency, ranges)
    show_fft_frequency(filtered, samplingFrequency, ranges)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6), sharex=True)
    ax1.plot(t[frame_range[0]:frame_range[1]], sig[frame_range[0]:frame_range[1]], marker='v')

    ax2.plot(t[frame_range[0]:frame_range[1]], filtered[frame_range[0]:frame_range[1]], marker='v')
    ax2.set_xlabel('Frame')
    plt.tight_layout()
    plt.show()
    return filtered

def show_metrics(data, ranges, plot_ranges):
    
    len_img = 16
    img_per_row = 8
    fig, ax = plt.subplots(len_img//img_per_row+1*int(len_img%img_per_row>0), img_per_row, figsize=(16, 2*len_img//img_per_row+1))
    for i in range(len_img):
        ax[i//img_per_row, i%img_per_row].title.set_text(i)
        if ranges:
            ax[i//img_per_row, i%img_per_row].imshow(data[i][ranges[0]:ranges[1], ranges[2]:ranges[3]])
        else:
            ax[i//img_per_row, i%img_per_row].imshow(data[i])
            
    plt.show()

    sum_list, max_list, min_list, mean_list, std_list = [], [], [], [], []
    for i in range(len(data)):
        if ranges:
            img = data[i][ranges[0]:ranges[1], ranges[2]:ranges[3]]
        else:
            img = data[i]
        sum_list.append(np.sum(img))
        max_list.append(np.max(img))
        min_list.append(np.min(img))
        mean_list.append(np.mean(img))
        std_list.append(np.std(img))
    
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
       
    if ranges:
        h = ax[0, 0].plot(sum_list[plot_ranges[0]:plot_ranges[1]])
        ax[0, 0].title.set_text('sum_list')
        
        h = ax[0, 1].plot(max_list[plot_ranges[0]:plot_ranges[1]])
        ax[0, 1].title.set_text('max_list')
        
        h = ax[1, 0].plot(min_list[plot_ranges[0]:plot_ranges[1]])
        ax[1, 0].title.set_text('min_list')

        h = ax[1, 1].plot(mean_list[plot_ranges[0]:plot_ranges[1]])
        ax[1, 1].title.set_text('mean_list')

        h = ax[2, 0].plot(std_list[plot_ranges[0]:plot_ranges[1]])
        ax[2, 0].title.set_text('std_list')

    
    else:
        h = ax[0, 0].plot(sum_list)
        ax[0, 0].title.set_text('sum_list')
        
        h = ax[0, 1].plot(max_list)
        ax[0, 1].title.set_text('max_list')

        h = ax[1, 0].plot(min_list)
        ax[1, 0].title.set_text('min_list')

        h = ax[1, 1].plot(mean_list)
        ax[1, 1].title.set_text('mean_list')

        h = ax[2, 0].plot(std_list)
        ax[2, 0].title.set_text('std_list')

    plt.show()
    return sum_list, max_list, min_list, mean_list, std_list


# add referece for Josh's repository
class Gaussian():
    def __init__(self):
        self.a = 0

    def gaussian(self, height, center_x, center_y, width_x, width_y, rotation):
        """Returns a gaussian function with the given parameters"""
        
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)
        
        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2+
                  ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y, 0.0

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
    
    def recreate_gaussian(self, image):
        para = self.fitgaussian(image)
        y = np.linspace(0, image.shape[0], image.shape[0])
        x = np.linspace(0, image.shape[1], image.shape[1])
        x, y = np.meshgrid(x, y)
        return self.gaussian(*para)(y, x), para
    
# def generate_parameter_file(RHEED_spot_file, Parameters_file, growth_titles, spot_p, show=True):
    
#     if os.path.isfile(Parameters_file): os.remove(Parameters_file)

#     h5_para = h5py.File(Parameters_file, mode='a')

#     spots_title = ['spot_1', 'spot_2', 'spot_3']
#     para_title = ['img_intensity', 'img_fit_intensity']

#     for k in growth_titles:
#         print(k)
#         h5_growth = h5_para.create_group(k)
#         with h5py.File(RHEED_spot_file, mode='r') as h5_from:
#             print(h5_from.keys())
#             spot_1 = NormalizeData(np.array(h5_from[k][:, spot_p[0,0]:spot_p[0,1], spot_p[0,2]:spot_p[0,3]]))
#             spot_2 = NormalizeData(np.array(h5_from[k][:, spot_p[1,0]:spot_p[1,1], spot_p[1,2]:spot_p[1,3]]))
#             spot_3 = NormalizeData(np.array(h5_from[k][:, spot_p[2,0]:spot_p[2,1], spot_p[2,2]:spot_p[2,3]]))
#         print(spot_1.shape, spot_2.shape, spot_3.shape)

#         para_np_all = []
#         spots = [spot_1, spot_2, spot_3]
#         for j, spot in enumerate(spots):
#             h5_spot = h5_growth.create_group(spots_title[j])
#             img_list, img_intensity_list, output_list, output_intensity_list = [], [], [], []

#             print('---------------spot', j, '----------------')
#             for i, img in enumerate(spot):
#                 output, para = Gaussian().recreate_gaussian(img)
                
#                 img_list.append(img)
#                 img_intensity_list.append(np.sum(img))
#                 output_list.append(output)
#                 output_intensity_list.append(np.sum(output))

#                 if show and (i==0 or i==len(spot)-1):
#                     sample_list = [img, output, output-img]
#                     labels = ['original', 'output', 'difference']
#                     show_images(sample_list, labels, img_per_row=3)

#             img_list = np.stack(img_list, axis=0)
#             img_intensity_list = np.array(img_intensity_list)
#             output_list = np.stack(output_list, axis=0)
#             output_intensity_list = np.array(output_intensity_list)

#             h5_spot.create_dataset('img', data=img_list)
#             h5_spot.create_dataset('img_intensity', data=img_intensity_list)
#             h5_spot.create_dataset('img_fit', data=output_list)
#             h5_spot.create_dataset('img_fit_intensity', data=output_intensity_list)

#     h5_para.close()
    
    


# class GenerateParameterFile:
#     def __init__(self, RHEED_image_dataset_path, growth_names, crop_dict):
#         '''
#         spots_names = ['spot_1', 'spot_2', 'spot_3']
#         '''
#         self.growth_names = growth_names
#         self.spots_names = list(crop_dict.keys())
#         self.crop_dict = crop_dict
#         self.RHEED_image_dataset_path = RHEED_image_dataset_path
    
#     def generate_parameter_file(self, parameters_file_path, func, cover_file=False, num_workers=1):
#         if os.path.isfile(parameters_file_path):
#             print('h5 file exist.')
#             if cover_file: 
#                 os.remove(parameters_file_path)
#                 print('Replace with new file.')
#         h5_file = h5py.File(self.RHEED_image_dataset_path, mode='r')

#         with h5py.File(parameters_file_path, mode='a') as h5_para:
#             for growth in self.growth_names:
#                 h5_growth = h5_para.create_group(growth)
#                 for spot in self.spots_names:
#                     h5_spot = h5_growth.create_group(spot)
#                     inputs = self.normalize_inputs(h5_file, growth, spot)

#                     if num_workers > 1:
#                         tasks = [delayed(self.fit)(func, img) for img in inputs]
#                         results = Parallel(n_jobs=num_workers)(tasks)
#                         img_all = np.array([res[0] for res in results])
#                         img_intensity_all = np.array([res[1] for res in results])
#                         img_rec_all = np.array([res[2] for res in results])
#                         img_rec_intensity_all = np.array([res[3] for res in results])

#                     else:
#                         results = [self.fit(func, img) for img in inputs]
#                         img_all = np.array([res[0] for res in results])
#                         img_intensity_all = np.array([res[1] for res in results])
#                         img_rec_all = np.array([res[2] for res in results])
#                         img_rec_intensity_all = np.array([res[3] for res in results])                            

#                     h5_spot.create_dataset('img', data=img_all)
#                     h5_spot.create_dataset('img_intensity', data=img_intensity_all)
#                     h5_spot.create_dataset('img_fit', data=img_rec_all)
#                     h5_spot.create_dataset('img_fit_intensity', data=img_rec_intensity_all)

#     def normalize_inputs(self, h5_file, growth, spot):
#         crop = self.crop_dict[spot]
#         inputs = NormalizeData(np.array(h5_file[growth][:, crop[0]:crop[1], crop[2]:crop[3]]))
#         return inputs
        
#     def fit(self, func, img):
#         img_rec, para = func(img)
#         img_intensity = np.sum(img)
#         img_rec_intensity = np.sum(img_rec)
#         return img, img_intensity, img_rec, img_rec_intensity
    
    
class RHEED_image_processer:
    def __init__(self, RHEED_image_dataset_path, crop_dict, fit_function, num_workers):
        '''
        spots_names = ['spot_1', 'spot_2', 'spot_3']
        '''
        self.crop_dict = crop_dict
        self.RHEED_image_dataset_path = RHEED_image_dataset_path
        self.fit_function = fit_function
        self.num_workers = num_workers
    
    def write_h5_file(self, parameters_file_path, growth_names, cover_file=False):
        '''
        parameters: img_sum, img_max, img_mean, img_rec_sum, img_rec_max, img_rec_mean, height, x, y, width_x, width_y
        '''
        spots_names = list(self.crop_dict.keys())
        
        if os.path.isfile(parameters_file_path):
            print('h5 file exist.')
            if cover_file: 
                os.remove(parameters_file_path)
                print('Replace with new file.')
        with h5py.File(self.RHEED_image_dataset_path, mode='r') as h5_file:
            with h5py.File(parameters_file_path, mode='a') as h5_para:
                for growth in growth_names:
                    h5_growth = h5_para.create_group(growth)
                    for spot in spots_names:
                        h5_spot = h5_growth.create_group(spot)
                        inputs = self.normalize_inputs(h5_file, growth, spot)
                        results = self.fit_batch(inputs)

                        img_all = np.array([res[0] for res in results])
                        img_rec_all = np.array([res[1] for res in results])
                        parameters = np.array([res[2] for res in results])

                        h5_spot.create_dataset('img', data=img_all)
                        h5_spot.create_dataset('img_rec', data=img_rec_all)
                        h5_spot.create_dataset('parameters', data=parameters)
                    
    def normalize_inputs(self, h5_file, growth, spot):
        crop = self.crop_dict[spot]
        inputs = NormalizeData(np.array(h5_file[growth][:, crop[0]:crop[1], crop[2]:crop[3]]))
        return inputs
        
    def fit_batch(self, inputs):
        if self.num_workers > 1:
            tasks = [delayed(self.fit)(img) for img in inputs]
            results = Parallel(n_jobs=self.num_workers)(tasks)
        else:
            results = [self.fit(img) for img in inputs]
        return results
            
    def fit(self, img):
        img_rec, para = self.fit_function(img) # para: height, x, y, width_x, width_y, 0.0
        img_sum, img_max, img_mean = np.sum(img), np.max(img), np.mean(img)
        img_rec_sum, img_rec_max, img_rec_mean = np.sum(img_rec), np.max(img_rec), np.mean(img_rec)
        parameters = [img_sum, img_max, img_mean, img_rec_sum, img_rec_max, img_rec_mean, *para]
        return img, img_rec, parameters
    
    def visualize(self, growth, spot, frame):
        with h5py.File(self.RHEED_image_dataset_path, mode='r') as h5_file:
            img = self.normalize_inputs(h5_file, growth, spot)[frame]
            img, img_rec, parameters = self.fit(img)
        print(f'img_sum:{parameters[0]}, img_max:{parameters[1]}, img_mean:{parameters[2]}')
        print(f'img_rec_sum:{parameters[3]}, img_rec_max:{parameters[4]}, img_rec_mean:{parameters[5]}')
        print(f'height:{parameters[6]}, x:{parameters[7]}, y:{parameters[8]}, width_x:{parameters[9]}, width_y_max:{parameters[10]}')
        sample_list = [img, img_rec, img_rec-img]
        labels = ['original', 'output', 'difference']
        show_images(sample_list, labels, img_per_row=3)
        return img, img_rec, parameters