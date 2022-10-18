
import os, glob, h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab as pl
import matplotlib
import matplotlib.ticker as ticker
import matplotlib as mpl
import seaborn as sns

from scipy.signal import butter, lfilter, sosfilt, freqz
from scipy import stats
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import zscore
import pickle

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