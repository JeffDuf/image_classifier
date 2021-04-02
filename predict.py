
import argparse as ap
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict 
import json
import os
from datetime import datetime
import ImageClassifierProject as cl
from PIL import Image
import numpy as np

# python predict.py flowers/valid/43/image_02330.jpg saves/SavedTrainingModel_resnet18_2021_04_01-05:58:16_PM_accuracy_77_accuracy_81.pth --category_names cat_to_name.json --top_k 5 --gpu

# Parse arguments
parser = ap.ArgumentParser()
parser.add_argument("image_path", type=str, default='flowers/valid/43/image_02330.jpg', help='Filepath of the image to use for predictition. E.g.: flowers/valid/43/image_02330.jpg')
parser.add_argument("checkpoint", type=str, default='saves/SavedTrainingModel_resnet18_2021_04_01-05:58:16_PM_accuracy_77.pth', help='Saved model checkpoint to use for the prediction.')
parser.add_argument("--category_names", type=str, default='cat_to_name.json', help='JSON list mapping class ID number and the class description in words.')
parser.add_argument("--top_k", type=int, default=3, help='Provides the n classes with the highest probability. E.g.: 5, will return the top 5 contenders.')
parser.add_argument("--gpu", default='cpu', nargs='?', const='', help='If available, use the GPU instead of the CPU. Do not set argument if no GPU is needed.')
cli_args = parser.parse_args()
print(cli_args)
image_path = cli_args.image_path
checkpoint_path = cli_args.checkpoint
top_k = cli_args.top_k
category_names = cli_args.category_names
arg_gpu = True
if cli_args.gpu == "cpu":
    arg_gpu = False
device = "cpu"
if arg_gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing a single prediction   
model, current_epoch = cl.load_checkpoint(checkpoint_path)
flowers_filenames = cl.load_classes_names (category_names)
top_classes_probs, top_classes_IDs, image = cl.predict (image_path, model, device, top_k)

# Analyze the data
top_class_names = []
for top_key_i in range(len(top_classes_IDs)):
    top_key = str(top_classes_IDs[top_key_i])
    top_prob = int(round(float(top_classes_probs[top_key_i])*100))
    top_class_names.append(str(flowers_filenames[top_key]) + " {} ".format(top_key))
    print("flower ID = {}, flower = {}, probability of {}%".format(top_key,flowers_filenames[top_key], top_prob))
    
    
    
    