from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def check_cropped():
    line_start = (135, 250)  
    line_end = (150, 265) 
    line_pixels = np.array(image.crop((line_start[0], line_start[1], line_end[0], line_end[1])))
    plt.grid(True)
    plt.imshow(line_pixels)
    
def horizontal():
    line_start = (345, 130)  
    line_end = (480, 190) 
    ylim = 120
    return line_start, line_end, ylim

def tilt(): 
    line_start = (135, 250)  
    line_end = (150, 265) 
    ylim = 12.5
    return line_start, line_end, ylim
    
selection = 'horizontal' # option: horizontal, tilt

if selection == 'horizontal': 
    line_start, line_end, ylim = horizontal()
else:
    line_start, line_end, ylim = tilt()

for i in [0, 0.5, 0.75, 1]:
    image = Image.open(f"blurred-input/{selection}/blurred-input-5x5-{i:.2f}.png").convert('L')
    
    line_pixels = image.crop((line_start[0], line_start[1], line_end[0], line_end[1]))
    hist = line_pixels.histogram()

    plt.clf()
    plt.figure()
    plt.title('Image Histogram of Cropped Region', fontsize=20)
    plt.xlabel('Pixel Value', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, ylim)
    plt.plot(hist[:-1])
    plt.savefig(f"blurred-input/{selection}-cropped-local-{i:.2f}.png", bbox_inches='tight', pad_inches = 0, dpi=300)