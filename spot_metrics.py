import numpy as np
import matplotlib.pyplot as plt

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