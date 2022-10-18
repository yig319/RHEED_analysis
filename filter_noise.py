import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, sosfilt, freqz
    

def show_sound(sound, sound_fitted, colored_data, peaks, xlabel, ylabel, title, save_path=None):
    sound = np.array(sound)
    if len(sound.shape) == 1:
        sound = np.copy(np.expand_dims(sound, axis=1))
        
    fig, ax = plt.subplots(1, 1)
    plt.plot(sound, marker='.', markersize=1, zorder=0, color='k')
    
    if type(sound_fitted) != type(None):
        ax.plot(sound_fitted, zorder=5)

    # add text for peaks
    if type(peaks) != type(None):
        for x, y in zip(peaks, sound[peaks]):
            pl.text(x, y, str(x), color="red", fontsize=6)
            
    if type(colored_data) != type(None):
        ax.scatter(colored_data[0], colored_data[1], zorder=10)
        ax.scatter(colored_data[2], colored_data[3], zorder=15)   
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
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

