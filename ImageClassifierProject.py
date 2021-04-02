#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Package Imports:
#  - All the necessary packages and modules are imported in the first cell of the notebook
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict 
import json
import os
from datetime import datetime
from PIL import Image
import numpy as np

# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


# Training data augmentation: 
#   - torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
# Data normalization:
#   - The training, validation, and testing data is appropriately cropped and normalized
def get_transforms ():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return train_transforms, test_transforms, validation_transforms


# In[3]:


# Data Loading
#   - The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
def get_datasets (train_transforms, test_transforms, validation_transforms, data_dir = 'flowers'):    
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    return train_data, test_data, validation_data


# In[4]:


# Data batching:
#   - The data for each set is loaded with torchvision's DataLoader
def get_trainloaders (datadir = 'flowers'):

    train_transforms, test_transforms, validation_transforms = get_transforms ()
    train_data, test_data, validation_data = get_datasets (train_transforms, test_transforms, validation_transforms, datadir)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    
    return trainloader, testloader, validationloader, train_data


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


def load_classes_names (filepath):
    with open(filepath, 'r') as f:
        flowers_filenames = json.load(f)   
    return flowers_filenames


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[91]:


# Pretrained Network
#   - A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
def get_std_resnet18_model ():
    # Build and train your network
   # model = models.resnet18(pretrained = True)

    # Original resnet18 Classifier:     (fc): Linear(in_features=512, out_features=1000, bias=True)
    # Original vgg19 Classifier:        (fc): Linear(in_features=25088, out_features=1000, bias=True)
    # Original vgg11 Classifier:          (classifier): Sequential(
                                        #    (0): Linear(in_features=25088, out_features=4096, bias=True)
                                        #    (1): ReLU(inplace)
                                        #    (2): Dropout(p=0.5)
                                        #    (3): Linear(in_features=4096, out_features=4096, bias=True)
                                        #    (4): ReLU(inplace)
                                        #    (5): Dropout(p=0.5)
                                        #    (6): Linear(in_features=4096, out_features=1000, bias=True)
    # Original densenet201 Classifier:  (classifier): Linear(in_features=1920, out_features=1000, bias=True)
    # Original resnet101 Classifier:    (fc): Linear(in_features=2048, out_features=1000, bias=True)
    return models.resnet18(pretrained = True)


# In[92]:


def get_std_densenet_model ():
    return models.densenet(pretrained = True)
def get_std_densenet121_model ():
    return models.densenet121(pretrained = True)
def get_std_densenet169_model ():
    return models.densenet169(pretrained = True)
def get_std_densenet161_model ():
    return models.densenet167(pretrained = True)
def get_std_densenet201_model ():
    return models.densenet201(pretrained = True)
def get_std_inception_model ():
    return models.densenet201(pretrained = True)
def get_std_alexnet_model ():
    return models.alexnet(pretrained = True)
def get_std_vgg_model ():
    return models.vgg(pretrained = True)
def get_std_vgg11_model ():
    return models.vgg11(pretrained = True)
def get_std_vgg13_model ():
    return models.vgg13(pretrained = True)
def get_std_vgg16_model ():
    return models.vgg16(pretrained = True)
def get_std_vgg19_model ():
    return models.vgg19(pretrained = True)
def get_std_resnet_model ():
    return models.resnet(pretrained = True)
def get_std_resnet101_model ():
    return models.resnet101(pretrained = True)
def get_std_resnet152_model ():
    return models.resnet152(pretrained = True)
def get_std_resnet34_model ():
    return models.resnet34(pretrained = True)
def get_std_resnet50_model ():
    return models.resnet50(pretrained = True)
def get_std_squeezenet_model ():
    return models.squeezenet(pretrained = True)


# In[93]:


# Feedforward Classifier
#   - A new feedforward network is defined for use as a classifier using the features as input
def get_classifier (model, nb_output_states, nb_hidden1_states, architecture):

    # We reuse the model above, so we can't change the inputs or the depth of the model.
    # What we can do is to replace the output layer by something else.
    # Here, we add 1 hidden layer and 1 output layer
    # The hidden layer connects to where the original model expected the output, therefore has 1024 inputs.
    # The new output layer has 102 flower types
    # We can decide, with 'nb_hidden1_states', how many states we want in the new hidden layer.
    # Note: We don't want to train the reused model, but we do want to train the new hidden/output layers.
    #nb_inputs_states = 512      
    #nb_hidden1_states = hidden_states
    #nb_output_states = 102
    
    #get_std_resnet18_model()   # 64 inputs, 512 classifier inputs
    #get_std_densenet121_model()# 64 inputs, 1024 classifier inputs
    #get_std_vgg11_model()      # 64 inputs, 25088 classifier inputs
    #get_std_alexnet_model()    # 64 inputs, 9216 classifier inputs


    # This is what the model expects for the classifier input. 
    for param in model.parameters(): # Freeze parameters so we don't backprop through them
        param.requires_grad = False
    if (architecture == "resnet18"):
        nb_inputs_states = 512
    elif  (architecture == "densenet121"):
        nb_inputs_states = 1024
    elif  (architecture == "vgg11"):
        nb_inputs_states = 25088
    else:#  (architecture == "alexnet"):
        nb_inputs_states = 9216

    

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear (nb_inputs_states, nb_hidden1_states)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc2', nn.Linear (nb_hidden1_states, nb_hidden1_states)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc3', nn.Linear (nb_hidden1_states, nb_output_states)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    return classifier


# In[94]:


# Helper method to return the criterion, for easy later-modification of the selected criterion
def get_criterion ():
    return nn.NLLLoss()


# In[95]:


# Helper method to return the optimizer, for easy later-modification of the selected optimizer
def get_optimizer (model, architecture, learning_rate = 0.003): 
    if (architecture == "resnet18"):
        params = model.fc.parameters()
    elif  (architecture == "densenet121"):
        params = model.classifier.parameters()
    elif  (architecture == "vgg11"):
        params = model.classifier.parameters()
    else:#  (architecture == "alexnet"):
        params = model.classifier.parameters()
    return optim.Adam(params, lr=learning_rate)


# In[96]:


# Testing Accuracy
#    - The network's accuracy is measured on the test data
# Note: made generic for either validation or testing.
def test_model_accuracy(model, device, image_verif_loader):
    test_loss = 0
    accuracy = 0
    criterion = get_criterion ()
    model.eval()
    with torch.no_grad():
        for inputs, labels in image_verif_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    model.train()
    return accuracy, test_loss


# In[97]:


# Helper method to print statistics about the training progress
def print_training_statistics (accuracy, test_loss, epochs_done, epochs_to_do, image_verif_loader):
    print("Epoch {}/{}..".format(epochs_done+1, epochs_to_do), 
          "Validation loss: {:.3f}.. ".format(test_loss/len(image_verif_loader)),
          "Test accuracy: {:.3f}".format(accuracy/len(image_verif_loader)))   


# In[98]:


# Training the network
#   - The parameters of the feedforward classifier are appropriately trained, 
#     while the parameters of the feature network are left static
#   - Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed
def train_model (model, architecture, device, train_loader, validation_loader, learning_rate, epochs_to_perform, print_each = 0, training_epochs_already_performed = 0):
    epochs = epochs_to_perform
    running_loss = 0
    steps = 0
    criterion = get_criterion ()
    optimizer = get_optimizer (model, architecture, learning_rate)
    
    for epoch in range(epochs):
        for images, targets in train_loader:   # len(images)=64, len(trainloader)=103    
                # Count the number of steps we went through across all epoch
            steps += 1
                # Convert to GPU if available
            images,targets = images.to(device), targets.to(device)   
                # Reset the optimizer at each pass, to avoid cumulative results
            optimizer.zero_grad()

            ###################################################
            # Make a forward pass
                # Make a pass forward in the model. logps is the log of the probabilities obtained at output
            logps = model.forward(images)
            #RuntimeError: size mismatch, m1: [64 x 512], m2: [1024 x 512] 
                # Calculate the loss in comparison to the targets
            loss = criterion (logps, targets)
            
            ###################################################
            # Make a backward pass to train the new classifier
                # Make back propagation pass
            loss.backward()
                # Perform an optimisation setup through the model
            optimizer.step()
                # Save the running loss, or cost, for the statistics at the end.
            running_loss =+ loss.item()

            if print_each == 0:
                continue
            
            # Each "print_each" pass, make a verification check of the progress.
            if steps % print_each == 0:
                epochs_done = training_epochs_already_performed+epoch #+1
                epochs_todo = training_epochs_already_performed+epochs
                validate(model, device, validation_loader, epochs_done, epochs_todo)
                running_loss = 0
    return running_loss     


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[99]:


# Validation on the test set

def validate (model, device, image_verif_loader, epochs_done = 0, epochs_todo = 0):
    accuracy, test_loss = test_model_accuracy(model, device, image_verif_loader)
    print_training_statistics(accuracy, test_loss, epochs_done, epochs_todo, image_verif_loader)
    return accuracy/len(image_verif_loader)*100


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[131]:


# Saving the model
#   - The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
def save_model_for_training (model, architecture, current_epoch, filename, traindata, learning_rate):
    model.to('cpu')
    model.class_to_idx = traindata.class_to_idx
    optimizer = get_optimizer (model, architecture, learning_rate)

    classifier_state_dict = ""
    if (architecture == "resnet18"):
        classifier = model.fc
    elif  (architecture == "densenet121"):
        classifier = model.classifier
        classifier_state_dict = classifier.state_dict
    elif  (architecture == "vgg11"):
        classifier = model.classifier
        classifier_state_dict = classifier.state_dict
    else :# (architecture == "alexnet"):
        classifier = model.classifier
        classifier_state_dict = classifier.state_dict

    checkpoint = {
        'training_epoch': current_epoch,
        'optimizer': optimizer,
        'arch': architecture,
        'classifier': classifier,
     #   'classifier_state_dict': classifier.state_dict,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': traindata.class_to_idx
    }
    torch.save(checkpoint, filename)


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[132]:


# Loading checkpoints
#   - There is a function that successfully loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    architecture = checkpoint['arch']
    model = getattr(models, architecture)(pretrained=True)
    #model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.arch = checkpoint['arch']
    
    if (architecture == "resnet18"):
        model.fc = checkpoint["classifier"]
    elif  (architecture == "densenet121"):
        model.classifier = checkpoint["classifier"]
        #model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    elif  (architecture == "vgg11"):
        model.classifier = checkpoint["classifier"]
       # model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:#architecture == "alexnet"):
        model.classifier = checkpoint["classifier"] 
        #model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
   
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    current_epoch = int(checkpoint['training_epoch'])
    
    return model, current_epoch


# In[102]:


def get_model(architecture):
    if (architecture == "resnet18"):
        return get_std_resnet18_model()
    elif  (architecture == "densenet121"):
        return get_std_densenet121_model()
    elif  (architecture == "vgg11"):
        return get_std_vgg11_model()
    else:#  (architecture == "alexnet"):
        return get_std_alexnet_model()
   # return "Invalid model architecture"


# In[103]:


# Start from Scratch, no loading of model
def train_from_scratch (architecture, output_states, hiddenstates):
    model = get_model (architecture)
    
    if (architecture == "resnet18"):
        model.fc = get_classifier (model, output_states, hiddenstates, architecture)
    elif  (architecture == "densenet121"):
        model.classifier = get_classifier (model, output_states, hiddenstates, architecture)
    elif  (architecture == "vgg11"):
        model.classifier = get_classifier (model, output_states, hiddenstates, architecture)
    else:#  (architecture == "alexnet"):
        model.classifier = get_classifier (model, output_states, hiddenstates, architecture)
   

    return model


# In[104]:


def get_nb_output_classes (data_dir):
    nb_output_classes = 0

    for _, dirnames, filenames in os.walk(data_dir + "/train"):
      # ^ this idiom means "we won't be using this value"
        nb_output_classes += len(dirnames)

    return nb_output_classes





























# Image Processing
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 
# This can be done with the thumbnail or resize methods.
def image_thumbnail (image, size = 256):  
    # Resize to 256 x 256
#    width, height = image.size   # Get dimensions
#    image.thumbnail((256,256))
#    width, height = image.size   # Get dimensions
    
    
    width, height = image.size
    aspect_ratio = width / height
    if width < height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    elif height < width:
        new_height = size
        new_width = int(width * aspect_ratio)
    else: # when both sides are equal
        new_width = size
        new_height = size
    image = image.resize((new_width, new_height))
    
    return image

# Image Processing
# Crop out the center 224x224 portion of the image.
def image_crop_center (image, size = 224):
    width, height = image.size   # Get dimensions
    new_width = new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2    
    image = image.crop((left, top, right, bottom)) # Crop the center of the image
        
    width, height = image.size   # Get dimensions    
    #print("Cropped image size = {}x{}".format(width, height))
    return image

# Image Processing
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
def image_color_reencode (image):            
    image = np.array(image)
    #print("Image encoded to Min:{} Max:{}".format(image.min(), image.max()))
    image = image / 255
    #print("Image re-encoded to Min:{} Max:{}".format(image.min(), image.max()))

    return image

# Image Processing
# As before, the network expects the images to be normalized in a specific way. 
# For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. 
# You'll want to subtract the means from each color channel, then divide by the standard deviation.
def image_color_normalize (image):
    means = np.array([0.485, 0.456, 0.406])
    sds = np.array([0.229, 0.224, 0.225])
    
    #print("Image to be normalized with means:{} std_dev:{}".format(means, sds))
    image = (image - means) / sds  
    #print("Image normalized to Min:{} Max:{}".format(image.min(), image.max(), ))
    
    return image


# Image Processing
# And finally, PyTorch expects the color channel to be the first dimension but 
# it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose.
# The color channel needs to be first and retain the order of the other two dimensions.
def image_transpose (image):
    image = image.transpose((2, 0, 1))  
    return image


# Image Processing:
#    - The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
def process_image(image):
    image = image_thumbnail (image)
    image = image_crop_center (image)
    image = image_color_reencode (image)
    image = image_color_normalize (image)
    image = image_transpose (image)
    return image


# Class Prediction:
#   - The predict function successfully takes the path to an image and a checkpoint, 
#     then returns the top K most probably classes for that image

def predict(image_path, model, device, topk=5):
    image = torch.from_numpy(process_image(Image.open(image_path)))
    image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor);
    image = image.to(device)
    
    criterion = get_criterion ()
    model.eval()
    model.to(device)
    with torch.no_grad():        
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_class = ps.topk(topk, dim=1)        
    model.train();
    
    # Note: top_class returns the array of indexes, not of classes.
    probs_np = top_class[0][0].cpu().numpy()
    classes_idx = top_class[1][0].cpu().numpy()
    classes = []
    for class_idx in classes_idx:
        classes.append(get_key_from__class_to_idx (model.class_to_idx, class_idx))
    
    return probs_np, classes, image


# helper to retrieve the key from the index. 
# Note: will add exception handling later.
def get_key_from__class_to_idx(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key




