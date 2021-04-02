
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

# python train.py flowers --save_dir saves --arch resnet18 --learning_rate 0.003 --hidden_units 512 --epochs 5 --gpu --checkpoint saves/SavedTrainingModel_resnet18_2021_04_01-05:58:16_PM_accuracy_77_accuracy_81.pth

# Parse arguments
parser = ap.ArgumentParser()
parser.add_argument("data_dir", type=str, default='flowers', help='Path to the train, test and valid folders containing the class folders. E.g.: flowers')
parser.add_argument("--save_dir", type=str, default='saves', help='Path to save the model. E.g.: saves')
parser.add_argument("--arch", type=str, default='resnet18', help='Selected model. Supported models are: resnet18, vgg11, densenet121, alexnet.')
parser.add_argument("--learning_rate", type=float, default=0.003, help='Learning rate to avoid big jumps in the gradients. E.g.: 0.003')
parser.add_argument("--hidden_units", type=int, default=512, help='Number of states in the hidden layer of the classifier. E.g.: 128')
parser.add_argument("--epochs", type=int, default = 1, help='Number of epochs to train. E.g. 5')
parser.add_argument("--gpu", default='cpu', nargs='?', const='', help='If available, use the GPU instead of the CPU. Do not set argument if no GPU is needed.')
parser.add_argument("--checkpoint", type=str, default='None', help='Provide an existing checkpoint filepath to continue the training. E.g. saves/checkpoint.pth')

cli_args = parser.parse_args()
print(cli_args)

arg_arch = cli_args.arch
arg_data_dir = cli_args.data_dir
arg_save_dir = cli_args.save_dir
arg_learning_rate = cli_args.learning_rate
arg_epochs = cli_args.epochs
arg_hidden_states = cli_args.hidden_units
arg_use_checkpoint = cli_args.checkpoint
arg_gpu = True
if cli_args.gpu == "cpu":
    arg_gpu = False

datadir = arg_data_dir
learning_rate = arg_learning_rate
epoch_to_train = arg_epochs
architecture = arg_arch
hidden_states = arg_hidden_states
savedir = arg_save_dir
nb_output_classes = cl.get_nb_output_classes (datadir)

# Ensure there is a folder to save, if not, create it.
try:
    os.mkdir(savedir)
except OSError:
    pass

# Choose the device. Select GPU if requested and if available.
device = "cpu"
if arg_gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare filenames to save
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
stored_filename = savedir + f"/SavedTrainingModel_{architecture}_{date}.pth"
current_epoch = 0
trainloader, testloader, validationloader, train_data = cl.get_trainloaders (datadir)

# If checkpoint provided in argument, continue the training. If not, start from scratch
if arg_use_checkpoint != "None":
    stored_filename = arg_use_checkpoint
    model, current_epoch = cl.load_checkpoint(stored_filename)
else:
    model = cl.train_from_scratch (architecture, nb_output_classes, hidden_states)

model.to(device); # Store model in GPU memory if available


print_each = 25
#Run a few more iterations on training the model
print ("Starting training the model with: device = {}, architecture = {}...".format(device, architecture))
print (f"--> epochs_to_perform = {epoch_to_train}, print_each = {print_each} steps...")
print ("Periodically validating the model's progress (each {} steps) with the validation_dataloader...".format (print_each))
cl.train_model(model, architecture, device, trainloader, validationloader, learning_rate, epoch_to_train, print_each, training_epochs_already_performed = current_epoch)
current_epoch += epoch_to_train

print ("Testing the model with the test_dataloader...")
accuracy_percent = cl.validate (model, device, testloader, current_epoch, current_epoch)
stored_filename = stored_filename.replace (".pth", "_accuracy_" + str(int(round(accuracy_percent))) + ".pth")
cl.save_model_for_training(model, architecture, current_epoch, stored_filename, train_data, learning_rate)
print ("Model saved under {} after {} epochs.".format(stored_filename, current_epoch))