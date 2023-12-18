import cv2
import os
import io
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import skimage.filters
import skimage.measure

h,w= 128, 192

def set_device(model):
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model = model.to(device)
     return model, device


def show_img_mask(img, mask):
    if torch.is_tensor(img):
        img=to_pil_image(img)
        mask=to_pil_image(mask)
        
    img_mask=mark_boundaries(np.array(img), np.array(mask), outline_color=(0,1,0),color=(0,1,0))

    plt.imshow(img_mask)
    return img_mask

def processImage(img, device, mod):

    image = Image.open(img).convert('L')
    image = image.resize((w, h))
    img_t=to_tensor(image).unsqueeze(0).to(device)

    tumor_size_mm=0.0
    pred = mod(img_t).cpu()

    pred=torch.sigmoid(pred)[0]
    mask_pred = (pred[0]>=0.5)

    image_gray = mask_pred.numpy().astype('uint8') * 255  # Convert the mask to grayscale

    thresh = skimage.filters.threshold_otsu(image_gray)
    tumor_mask = image_gray > thresh
    
    pixel_size = 0.1  # the size of a pixel in millimeters
    label_image = skimage.measure.label(tumor_mask)
    region_props = skimage.measure.regionprops(label_image)
    
    if len(region_props) == 0:
        tumor_size_mm = 0.0
    else:
        tumor_area = region_props[0].area
        tumor_size_mm = tumor_area * pixel_size ** 2

    plt.figure()
    plt.subplot(1, 3, 1) 
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2) 
    plt.imshow(mask_pred, cmap="gray")
    
    plt.subplot(1, 3, 3) 
    res_img=show_img_mask(image, mask_pred)
    answer="The tumor size is "+str(tumor_size_mm)+" mm^2"
    return res_img, answer