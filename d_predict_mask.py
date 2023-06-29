from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib import colors
color = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow', 'blue'])
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor()])
from unet import UNet
import os

def load_model(model_path='run10=5.83e-7.pth'):
    model = UNet(n_channels=3, n_classes=5, bilinear=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval();
    return model

def visualise_predicted(model, image, kernel, sigma=0):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = nn.functional.softmax(output, dim=1)[0] 
        probs = probs.data.numpy().transpose((1, 2, 0))#.detach.cpu().numpy()
        mask = np.argmax(probs, axis=2)
        # Image.fromarray((mask).astype(np.uint8)).save(f"predicted_mask/blurred/-predicted-mask-k-{kernel}-s-{sigma:.2f}.png")
        
        # plt.imshow(mask, cmap=color)
        # plt.axis('off')
        # plt.savefig(f"predicted_mask/blurred-predicted-k-{kernel}-s-{sigma:.2f}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
        
        plt.imshow(np.array(image))
        plt.imshow(mask, cmap=color, alpha=0.6)
        plt.axis('off')
        plt.savefig(f"predicted_mask/blurred-overlay-k-{kernel}-s-{sigma:.2f}.png", bbox_inches='tight', pad_inches = 0, dpi=300)#
    return mask


model = load_model(model_path='../run10=5.83e-7.pth')#2023-06-13-run2/0.00613.pth

for r in [0.0, 0.25, 0.50, 0.75, 1.0]:    
    image = Image.open(f"blurred-input/blurred-input-5x5-{r:.2f}.png").convert('RGB')
    image = Image.open(f"blurred-input/blurred-input-k-3-s-0.50.png").convert('RGB')
    mask = visualise_predicted(model, image, , 0.5)



