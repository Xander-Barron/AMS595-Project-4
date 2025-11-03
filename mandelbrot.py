import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, threshold):

    #grid of complex numbers across [xmin, xmax] x [ymin, ymax]
    x, y = np.mgrid[xmin:xmax:width*1j, ymin:ymax:height*1j]
    #complex numbers
    c = x + 1j*y
    #initialize z
    z = np.zeros_like(c)

    #mask to see which points are in the set
    mask = np.ones(c.shape, dtype = bool)

    for i in range (max_iter):
        #keep only diverged points
        z[mask] = z[mask] ** 2 + c[mask]
        #update mask
        mask[np.abs(z) > threshold] = False

    #only return the points in the set through the mask
    return mask


def main():
    #set by the assignment
    width = 1000
    height = 1000
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 50
    threshold = 100

    mandelbrot_mask = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, threshold)

#from assignment
    plt.imshow(mandelbrot_mask.T, extent=[-2, 1, -1.5, 1.5])
    plt.gray()
    #axis and title
    plt.title("Mandelbrot Set")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.savefig("mandelbrot.png", dpi=300)
    plt.show()

#run the above code
if __name__ == "__main__":
    main()