import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from scipy.signal import savgol_filter

    
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

    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot_curve(curve_x, curve_y, curve_x_fit=None, curve_y_fit=None, labels_dict=None, bg_colors=None, plot_type='scatter', markersize=1, xlabel=None, ylabel=None, xlim=None, ylim=None, logscale=False, yaxis_style='sci', title=None, legend=None, figsize=(12,2.5), save_path=None):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if type(bg_colors) != type(None):
        x_coor = bg_colors[:,0]
        colors = bg_colors[:,1:]

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
    
    if plot_type == 'scatter':
        plt.scatter(x=curve_x, y=curve_y, c='k', s=markersize)
        if type(curve_y_fit) != type(None):
            if type(curve_x_fit) != type(None):
                plt.scatter(x=curve_x_fit, y=curve_y_fit, c='r', s=markersize)
            else:
                plt.scatter(x=curve_x, y=curve_y_fit, c='r', s=markersize)

    if plot_type == 'lineplot':
        plt.plot(curve_x, curve_y, color='k', marker='.', markersize=markersize)
        if type(curve_y_fit) != type(None):
            if type(curve_x_fit) != type(None):
                plt.plot(curve_x_fit, curve_y_fit, color='b', marker='.', markersize=markersize)
            else:
                plt.plot(curve_x, curve_y_fit, color='b', marker='.', markersize=markersize)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if type(xlim) != type(None): ax.set_xlim(xlim)
    if type(ylim) != type(None): ax.set_ylim(ylim)
    if yaxis_style == 'sci':
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
    if type(labels_dict) != type(None):
        for x in labels_dict.keys():
            y = curve_y[np.where(curve_x==find_nearest(curve_x, x))]
            pl.text(x, y, str(labels_dict[x]), color="g", fontsize=6)
            
    if logscale: plt.yscale("log") 
    if legend: plt.legend(legend)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

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


            

"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""
from scipy import special
from matplotlib import animation, colors, ticker, path, patches, patheffects

Path = path.Path
PathPatch = patches.PathPatch
erf = special.erf
cmap = plt.get_cmap('viridis')

def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure
    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # extracts the vertices used to construct the path
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    #  makes a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    # adds path to axes
    axes.add_patch(pathpatch)

def scalebar(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds scalebar to figures
    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    # gets the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(
        np.int32(np.floor(x_lim[1] - x_lim[0]))), np.abs(np.int32(np.floor(y_lim[1] - y_lim[0])))
    # computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1],
                          np.int32(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1],
                          np.int32(np.floor(image_size)))

    # sets the location of the scalebar"
    if loc == 'br':
        x_start = x_point[np.int32(.9 * image_size // 1)]
        x_end = x_point[np.int32((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int32(.1 * image_size // 1)]
        y_end = y_point[np.int32((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int32((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int32(.9 * image_size // 1)]
        x_end = x_point[np.int32((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int32(.9 * image_size // 1)]
        y_end = y_point[np.int32((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int32((.9 - .075) * image_size // 1)]

    # makes the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    # adds the text label for the scalebar
    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])
    
    
def visualize_afm_image(img, colorbar_range=[0,0], scalebar_dict=None, filename=None):

    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    im = ax.imshow(img)
    scalebar(ax, image_size=scalebar_dict['image_size'], scale_size=scalebar_dict['scale_size'], 
             units=scalebar_dict['units'], loc='br')

    im.set_clim(colorbar_range) 
    fig.colorbar(im, ax=ax)
    
    ax.tick_params(which='both', bottom=False, left=False, right=False, top=False, labelbottom=False)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()
    
    
class afm_miscut():
    def __init__(self, image):
        self.image = image
        
    def rotate(self, angle, rot_scale, colorbar_range, prominence):
        image_rot = imutils.rotate(self.image, angle=angle, scale=rot_scale)
        visualize_afm_image(image_rot, colorbar_range=colorbar_range)
        
        y = image_rot[:, random.randrange(1*len(image_rot)//3, 2*len(image_rot)//3)]
        x = np.linspace(0, len(y)-1, len(y))
        peak_indices = signal.find_peaks(y, prominence=prominence)[0]
        valley_indices = signal.find_peaks(-y, prominence=prominence)[0]
        xp, yp, xv, yv = [], [], [], []
        for i in peak_indices:
            xp.append(x[i])
            yp.append(y[i])
        for i in valley_indices:
            xv.append(x[i])    
            yv.append(y[i]) 
        label_peaks(y, peak_indices, valley_indices)
        

    def line_miscut(self, x, y, elevation, prominence, visualize=False):
        peak_indices = signal.find_peaks(y, prominence=prominence)[0]
        valley_indices = signal.find_peaks(-y, prominence=prominence)[0]
#         print(peak_indices, valley_indices)
        if len(peak_indices)<3 or len(valley_indices)<3: # not enough peaks or vallies to average 
            return 0, 0, 0

        xp, yp, xv, yv = [], [], [], []
        for i in peak_indices:
            xp.append(x[i])
            yp.append(y[i])
        for i in valley_indices:
            xv.append(x[i])    
            yv.append(y[i])    

        dist, height, theta = [], [], []
        if elevation == 'before':
            if xp[0] > xv[0]: # fisrt point is valley
                xv = xv[1:]
                yv = yv[1:]
            for i in range(np.min([len(xp), len(xv)])):
                dist.append((xv[i]-xp[i]))
                height.append(yp[i]-yv[i])
                theta.append(math.degrees(math.atan(height[i]/dist[i])))

        if elevation == 'after':
            if xp[0] < xv[0]: # fisrt point is peak
                xp = xp[1:]
                yp = yp[1:]
            for i in range(np.min([len(xp), len(xv)])):
                dist.append((xp[i]-xv[i]))
                height.append(yp[i]-yv[i])
                theta.append(math.degrees(math.atan(height[i]/dist[i])))

        if visualize:
            label_peaks(y, peak_indices, valley_indices)
            print(f'In column {i}:')
            print(f'Average distance between step edges is: {np.mean(dist)}m')
            print(f'Average height change is: {np.mean(height)}m')    
            print(f'Average miscut angle is: {np.mean(theta)} degrees')  
        return np.mean(dist), np.mean(height), np.mean(theta)
        
    def image_miscut(self, actual_size, angle, rot_scale, step, elevation, prominence, visualize=False):
        image_rot = imutils.rotate(self.image, angle=angle, scale=rot_scale)
        if visualize:
            visualize_afm_image(image_rot, colorbar_range=colorbar_range)

        step_width, heights, miscut = [], [], []
        for i in np.arange(0, image_rot.shape[0], step):
            y = image_rot[:,i]
            x = np.linspace(0, actual_size/rot_scale, len(y))
            d, h, t = self.line_miscut(x, y, elevation, prominence, visualize=visualize)
#             print(d, h, t)
            if d != 0: # return 0 when not enough peaks to calculate
                step_width.append(d)
                heights.append(h)
                miscut.append(t)
        print(f'Average distance between step edges is: {np.mean(step_width)*1e6}μm')
        print(f'Average height change is: {np.mean(heights)*1e9}nm')    
        print(f'Average miscut angle is: {np.mean(miscut)} degrees') 
        return step_width, heights, miscut