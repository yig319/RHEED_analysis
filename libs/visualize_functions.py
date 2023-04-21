import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from scipy.signal import savgol_filter
import imutils

    
def make_fine_step(x, transparency, step, color, saturation=1, savgol_filter_level=(15,1)):
    x_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(x, x[1:])])
    
    transparency_FineStep = np.hstack([np.linspace(start, stop, num=step+1, endpoint=True)[:-1] for start, stop in zip(transparency, transparency[1:])])
    if not isinstance(savgol_filter_level, type(None)):
        transparency_FineStep_before = np.copy(transparency_FineStep)
        transparency_FineStep = savgol_filter(transparency_FineStep, savgol_filter_level[0]*step+1, savgol_filter_level[1])

    transparency_FineStep_norm = np.expand_dims((transparency_FineStep / max(transparency_FineStep)) * saturation, 1)
    transparency_FineStep_norm[transparency_FineStep_norm<0] = 0
    
    colors = np.repeat([[*color]], len(transparency_FineStep_norm), 0)
    colors_all = np.concatenate([colors, transparency_FineStep_norm], 1)
    return x_FineStep, colors_all
#     return x_FineStep, colors_all, transparency_FineStep_before, transparency_FineStep


def two_color_array(x_all, x1, x2, c1, c2, transparency=1):
    color_array = np.zeros([len(x_all), 4], dtype=np.float32)
    color_array[np.isin(x_all, x1)] = [*c1, transparency]
    color_array[np.isin(x_all, x2)] = [*c2, transparency]
    return color_array

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

    
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

def draw_background_colors(ax, bg_colors):
    if isinstance(bg_colors, tuple):
        ax.set_facecolor(bg_colors)
    elif bg_colors is not None:
        x_coor = bg_colors[:, 0]
        colors = bg_colors[:, 1:]
        for i in range(len(x_coor)):
            if i == 0:
                end = (x_coor[i] + x_coor[i+1]) / 2
                start = end - (x_coor[i+1] - x_coor[i])
            elif i == len(x_coor) - 1:
                start = (x_coor[i-1] + x_coor[i]) / 2
                end = start + (x_coor[i] - x_coor[i-1])
            else:
                start = (x_coor[i-1] + x_coor[i]) / 2
                end = (x_coor[i] + x_coor[i+1]) / 2
            ax.axvspan(start, end, facecolor=colors[i])
            
def draw_boxes(ax, boxes, box_color):
    for (box_start, box_end) in boxes:
        ax.axvspan(box_start, box_end, facecolor=box_color, edgecolor=box_color)

def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]

def plot_scatter(ax, curve_x, curve_y, color='k', markersize=1):
    ax.scatter(x=curve_x, y=curve_y, c=color, s=markersize)

def plot_lineplot(ax, curve_x, curve_y, color='k', markersize=1):
    ax.plot(curve_x, curve_y, color=color, marker='.', markersize=markersize)

def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', 
               logscale=False, legend=None):
    if type(xlabel) != type(None): ax.set_xlabel(xlabel)
    if type(ylabel) != type(None): ax.set_ylabel(ylabel)
    if type(title) != type(None): ax.set_title(title)
    if type(xlim) != type(None): ax.set_xlim(xlim)
    if type(ylim) != type(None): ax.set_ylim(ylim)
    if yaxis_style == 'sci':
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
    if logscale: plt.yscale("log") 
    if legend: plt.legend(legend)

def label_curves(ax, curve_x, curve_y, labels_dict):
    if type(labels_dict) != type(None):
        for x in labels_dict.keys():
            y = curve_y[np.where(curve_x==find_nearest(curve_x, x))]
            pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
            
            
def plot_curve(curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, plot_colors=['k', 'r'], plot_type='scatter', markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci', title=None, legend=None, figsize=(12,2.5), save_path=None):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if plot_type == 'scatter':
        plot_scatter(ax, curve_x, curve_y, plot_colors[0], markersize)
        if not isinstance(curve_y_fit, type(None)):
            if not isinstance(curve_x_fit, type(None)):
                plot_scatter(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
            else:
                plot_scatter(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                
    if plot_type == 'lineplot':
        plot_lineplot(ax, curve_x, curve_y, plot_colors[0], markersize)
        if not isinstance(curve_y_fit, type(None)):
            if not isinstance(curve_x_fit, type(None)):
                plot_lineplot(ax, curve_x_fit, curve_y_fit, plot_colors[1], markersize)
            else:
                plot_lineplot(ax, curve_x, curve_y_fit, plot_colors[1], markersize)
                
    set_labels(ax, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, yaxis_style=yaxis_style, 
               logscale=logscale, legend=legend)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
# def plot_curve(curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, plot_colors=['k', 'r'], labels_dict=None, bg_colors=None, boxes=None, box_color=None, plot_type='scatter', markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci', title=None, legend=None, figsize=(12,2.5), save_path=None):
    
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
    
#     if isinstance(bg_colors,  tuple):
#         ax.set_facecolor(bg_colors)
#     elif not isinstance(bg_colors,  type(None)):
#         x_coor = bg_colors[:,0]
#         colors = bg_colors[:,1:]
#         for i in range(len(x_coor)):
#             if i == 0: 
#                 end = (x_coor[i] + x_coor[i+1]) / 2
#                 start = end - (x_coor[i+1] - x_coor[i])
#             elif i == len(x_coor) - 1: 
#                 start = (x_coor[i-1] + x_coor[i]) / 2
#                 end = start + (x_coor[i] - x_coor[i-1])
#             else:
#                 start = (x_coor[i-1] + x_coor[i]) / 2
#                 end = (x_coor[i] + x_coor[i+1]) / 2
#             ax.axvspan(start, end, facecolor=colors[i])
            
#     if not isinstance(boxes,  type(None)):
#         for (box_start, box_end) in boxes:
#             ax.axvspan(box_start, box_end, facecolor=box_color, edgecolor=box_color)

#     if plot_type == 'scatter':
#         plt.scatter(x=curve_x, y=curve_y, c=plot_colors[0], s=markersize)
#         if type(curve_y_fit) != type(None):
#             if type(curve_x_fit) != type(None):
#                 plt.scatter(x=curve_x_fit, y=curve_y_fit, c=plot_colors[1], s=markersize)
#             else:
#                 plt.scatter(x=curve_x, y=curve_y_fit, c=plot_colors[1], s=markersize)

#     if plot_type == 'lineplot':
#         plt.plot(curve_x, curve_y, color=plot_colors[0], marker='.', markersize=markersize)
#         if type(curve_y_fit) != type(None):
#             if type(curve_x_fit) != type(None):
#                 plt.plot(curve_x_fit, curve_y_fit, color=plot_colors[1], marker='.', markersize=markersize)
#             else:
#                 plt.plot(curve_x, curve_y_fit, color=plot_colors[1], marker='.', markersize=markersize)

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     if type(xlim) != type(None): ax.set_xlim(xlim)
#     if type(ylim) != type(None): ax.set_ylim(ylim)
#     if yaxis_style == 'sci':
#         plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
#     if type(labels_dict) != type(None):
#         for x in labels_dict.keys():
#             y = curve_y[np.where(curve_x==find_nearest(curve_x, x))]
#             pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
            
#     if logscale: plt.yscale("log") 
#     if legend: plt.legend(legend)
#     if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

    
def show_grid_plots(xs, ys, labels=None, ys_fit1=None, ys_fit2=None, img_per_row=4, subplot_height=3, ylim=None, legend=None):

    if type(labels) == type(None): labels = range(len(ys))

    fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
                             figsize=(16, subplot_height*len(ys)//img_per_row+1))    
    trim_axes(axes, len(ys))
    
    for i in range(len(ys)):
        
        if len(ys) <= img_per_row:
            index = i%img_per_row
        else:
            index = (i//img_per_row), i%img_per_row
        
        im = axes[index].plot(xs[i], ys[i], marker='.', markersize=8, color=(44/255,123/255,182/255, 0.5))
        axes[index].title.set_text(labels[i])

        if type(ys_fit1) != type(None):
            im = axes[index].plot(xs[i], ys_fit1[i], linewidth=4, color=(217/255,95/255,2/255))
            
        if type(ys_fit2) != type(None):
            im = axes[index].plot(xs[i], ys_fit2[i], linewidth=2, color=(27/255,158/255,119/255))
            
            # green: (27/255,158/255,119/255), purple: (117/255,112/255,179/255)
            
        if type(ylim) != type(None):
            axes[index].set_ylim([ylim[0], ylim[1]])
            
        if type(legend) != type(None): axes[index].legend(legend)

    fig.tight_layout()
    plt.show()
    
    
    
import torch
import torch.nn as nn

def visualize_predictions(x_all, y_all, all_outputs):
    parameters_all, parameters_processed_all, x_coor_all, info = all_outputs
    [xs_all, ys_all, xs_processed_all, ys_processed_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, labels_all] = info
    
    x_coor_all = np.copy(x_coor_all)
    parameters = np.copy(parameters_all)

    # seperate RHEED data based on different function forms
    ktop = parameters_processed_all[:,-2:]
    x1_all, y1_all, x2_all, y2_all = [], [], [], []
    for k, xx, yy in zip(ktop, xs_all, ys_all):
        if k[0] == 0: 
            x1_all.append(xx)
            y1_all.append(yy)
        if k[0] == 1: 
            x2_all.append(xx)
            y2_all.append(yy)
    if x1_all != []: x1_all = np.concatenate(x1_all)
    if y1_all != []: y1_all = np.concatenate(y1_all)
    if x2_all != []: x2_all = np.concatenate(x2_all)
    if y2_all != []: y2_all = np.concatenate(y2_all)

    plot_curve(x_all, y_all, xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12,2.5), xlim=(-2, 135))

    plot_curve(x1_all, y1_all, x2_all, y2_all, xlabel='Time (s)', ylabel='Intensity (a.u.)', figsize=(12,2.5), xlim=(-2, 135))

    plot_curve(x_coor_all, parameters_processed_all[:,0], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: a (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_processed_all[:,1], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: b*x (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_processed_all[:,2], plot_type='lineplot', xlabel='Time (s)', ylabel='y1: c*x^2 (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))

    plot_curve(x_coor_all, parameters_processed_all[:,3], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m1 (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_processed_all[:,4], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: m2*x (a.u.)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))
    plot_curve(x_coor_all, parameters_processed_all[:,5], plot_type='lineplot', xlabel='Time (s)', ylabel='y2: Characteristic Time (s)', 
               yaxis_style='sci', figsize=(12, 4), xlim=(-2, 135))

    print('MSE loss for DL model fitting is:', nn.MSELoss()(torch.tensor(np.concatenate(ys_nor_all, 0)), 
                                                            torch.tensor(np.concatenate(ys_nor_fit_all, 0))).item())
    
    
    
    
def label_violinplot(ax, data, label_type='average', text_pos='center'):
    '''
    data: a list of list or numpy array for different 
    '''
    
    # Calculate number of obs per group & median to position labels
    xloc = range(len(data))
    yloc, text = [], [] 
    
    for i, d in enumerate(data):
        yloc.append(np.median(d))
        
        if label_type == 'number':
            text.append("n: "+str(len(d)))
        
        if label_type == 'average':
            text.append(str(round(np.median(d), 4)))

    for tick, label in zip(xloc, ax.get_xticklabels()):
        if text_pos == 'center':
            ax.text(xloc[tick], yloc[tick]*1.1, text[tick], horizontalalignment='center', size=14, weight='semibold')
        if text_pos == 'right':
            ax.text(xloc[tick]+0.02, yloc[tick]*0.7, text[tick], horizontalalignment='left', size=14, weight='semibold')


        