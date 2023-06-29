import cv2
import matplotlib.pyplot as plt

k = 5
for s in [0.25, 0.50, 0.75, 1.0]: 
    gaussian_kernel = cv2.getGaussianKernel(k, s)
    kernel_2D = gaussian_kernel @ gaussian_kernel.transpose() # @ is matrix multiplication operator
    plt.imshow(kernel_2D, cmap='hot')
    plt.axis('off')
    plt.savefig(f"gaussian_filter/kernel-{k}-sigma-{s:.2f}.png", bbox_inches='tight', pad_inches = 0, dpi=300)