import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import json

import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
masks_dir = "grad_cam_masks_matrix/"

model_list = {'resnet18':models.resnet18, 
                'resnet101': models.resnet101,
                'resnext50': models.resnext50_32x4d,
                'densenet121': models.densenet121,
                'mobilenet': models.mobilenet_v2,
                'vit': models.vit_b_16,
                'swin': models.swin_t,
                'inceptionv3': models.inception_v3,
                }

def wrap_model(model):
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def load_images(input_dir, batch_size):
    images = [] 
    filenames = []
    masks = []
    idx = 0
    for filepath in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir,filepath))
        image = image.resize((img_height, img_width)).convert('RGB')
        mask_path = os.path.join(masks_dir,filepath+'.npy')
        mask = np.load(mask_path)
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images.append(np.array(image).astype(np.float32)/255)
        filenames.append(os.path.basename(filepath))
        masks.append(mask)
        idx += 1
        if idx == batch_size:
            images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
            masks_array = np.array(masks)
            masks_tensor = torch.FloatTensor(masks_array).unsqueeze(1)
            
            yield filenames, images, masks_tensor # passing mask tensor
            filenames = []
            images = []
            idx = 0
    if idx > 0:
        images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
        masks_tensor = torch.FloatTensor(masks).unsqueeze(1)
        yield filenames, images, masks_tensor

def get_labels(names, f2l):
    labels = []
    for name in names:
        labels.append(f2l[name]-1)
    return torch.from_numpy(np.array(labels, dtype=np.int64))

def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_img(output_path,img):
    img = (img.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)

def save_json(data_dict, file_path="experimental_data.json"):
    with open(file_path, 'r') as file:
        results_data = json.load(file)

    results_data.append(data_dict)
    with open(file_path, "w") as json_file:
        json.dump(results_data, json_file, indent=1)