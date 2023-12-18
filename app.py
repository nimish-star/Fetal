from flask import Flask, render_template, request
import os
import cv2
import io
import base64
import numpy as np
import pickle
from greyscale import set_device, show_img_mask, processImage
import torchsummary
import matplotlib.pylab as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


class SegNet(nn.Module):
    def __init__(self, params):
        super(SegNet, self).__init__()
        
        C_in, H_in, W_in=params["input_shape"]
        init_f=params["initial_filters"] 
        num_outputs=params["num_outputs"] 

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(8*init_f, 16*init_f, kernel_size=3,padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = nn.Conv2d(16*init_f, 8*init_f, kernel_size=3,padding=1)
        self.conv_up2 = nn.Conv2d(8*init_f, 4*init_f, kernel_size=3,padding=1)
        self.conv_up3 = nn.Conv2d(4*init_f, 2*init_f, kernel_size=3,padding=1)
        self.conv_up4 = nn.Conv2d(2*init_f, init_f, kernel_size=3,padding=1)

        self.conv_out = nn.Conv2d(init_f, num_outputs , kernel_size=3,padding=1)    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)
        x = F.relu(x)

        x=self.upsample(x)
        x = self.conv_up1(x)
        x = F.relu(x)

        x=self.upsample(x)
        x = self.conv_up2(x)
        x = F.relu(x)
        
        x=self.upsample(x)
        x = self.conv_up3(x)
        x = F.relu(x)
        
        x=self.upsample(x)
        x = self.conv_up4(x)
        x = F.relu(x)

        x = self.conv_out(x)
        
        return x 


model = pickle.load( open('Fetal_model.pkl', 'rb'))
path2weights="weights.pt"
model.load_state_dict(torch.load(path2weights))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    image_file = request.files['image']

    if image_file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded image
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(upload_path)

    #Setting Device(GPU or CPU)
    mod, device = set_device(model)

    #Processing Uploaded Image
    result_image, statement = processImage(upload_path, device, mod)
    result_image_uint8 = (result_image * 255).astype(np.uint8)

    # Save the result image
    result_filename = 'result.png'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result_image_uint8, cv2.COLOR_RGB2BGR))

    # Encode the processed image as base64
    with open(result_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    #return render_template('index.html', result_image=encoded_image)
    return render_template('index.html', result_image=encoded_image, message=statement)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False,host='0.0.0.0')
