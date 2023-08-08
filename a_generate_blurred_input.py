from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

folder_name = "blurred-input"
if os.path.exists(folder_name) == False:
    os.mkdir(folder_name)
    
image = cv2.imread("f_01275.png", cv2.IMREAD_COLOR) # option1: f_01275 or f_01477 , option2: cv2.IMREAD_GRAYSCALE or cv2.IMREAD_UNCHANGED

for s in [0.25, 0.50, 0.75, 1.0]:
    blurred_image = cv2.GaussianBlur(image, (5,5), s) # parameter: image, kernel_size, sigma
    cv2.imwrite(f"{folder_name}/blurred-input-5x5-{s:.2f}.png", blurred_image)
    
# image histogram

for s in [0.50, 0.75, 1.0]:    
    image = Image.open(f"{folder_name}/horizontal/blurred-input-5x5-{s:.2f}.png").convert('L')
    image = Image.open("f_01275.png").convert('L')
    hist = image.histogram()

    norm = [float(i) / sum(hist) for i in hist]
    plt.clf()
    plt.figure()
    plt.title('Image Histogram', fontsize=20)
    plt.xlabel('Pixel Value', fontsize=18)
    plt.ylabel('Normalised Frequency', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 0.007)
    plt.plot(norm[:-1])
    plt.savefig(f"{folder_name}/blurred-hist-{s:.2f}.png", pad_inches = 0.1, bbox_inches='tight', dpi=300)
    
    cumulative_hist = np.cumsum(hist)
    cumulative_norm = [float(i) / sum(cumulative_hist) for i in cumulative_hist]
    plt.figure()
    plt.title('Cumulative Pixel Count', fontsize=20)
    plt.xlabel('Pixel Value', fontsize=18)
    plt.ylabel('Normalised Cumulative Count', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 0.0085)
    plt.plot(cumulative_norm[:-1])
    plt.savefig(f"{folder_name}/blurred-chist-{s:.2f}.png", pad_inches = 0.1, bbox_inches='tight', dpi=300)



