import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

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
            ax.text(xloc[tick], yloc[tick] + 0.03, text[tick], horizontalalignment='center', size=14, weight='semibold')
        if text_pos == 'left':
            ax.text(xloc[tick] + 0.02, yloc[tick] - 0.03, text[tick], horizontalalignment='left', size=14, weight='semibold')