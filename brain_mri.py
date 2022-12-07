# -*- coding: utf-8 -*-
# mount drive
from google.colab import drive
drive.mount('/content/drive')

# import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torchvision.transforms as transforms

"""# Import dataset"""

# set the path
BASE_DIR="/content/drive/MyDrive/Dataset/BRAINMRI/Dataset/normalized/"
# import the excel file
data_df=pd.read_excel('/content/drive/MyDrive/Dataset/BRAINMRI/Classification.xlsx')
# find the class
data_df["Class"]=data_df["Diagnosis"]>0
data_df.Class = data_df.Class.astype('int')
# print data
data_df.head()

# define the function for load images
class Milano_MRI_Dataset(Dataset):
  # __init__ is used to initialize the object's state
    def __init__(self,df:pd.DataFrame,imfolder:str,train:bool = True, transforms=None):
        self.df=df
        self.imfolder=imfolder
        self.train=train
        #self.transforms=transforms
     
     # Defines behavior for when an item is accessed, using the notation self[key] 
    def __getitem__(self,index):
      # get the each image   
        filename='w'+self.df.iloc[index]['ID_CF']+'_MRI_sMRI_'+self.df.iloc[index]['ID_CF']+'.nii.gz';
        # join the path
        # join one or more path components intelligently
        im_path=os.path.join(self.imfolder,filename)
        # load the path
        nimg = nb.load(im_path)
        x = np.array(nimg.dataobj)
        #x=nimg.get_fdata()
        # elements are in range 0 to 255. Set the custom Transforms to suit 
        x = transforms.ToTensor()(x)
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        x=x.unsqueeze(0).type(torch.FloatTensor);
        # find mean
        m=torch.mean(x)
        # find standard deviation
        s=torch.std(x)
        x=(x-m)/s
        #if(self.transforms):
        #    x=self.transforms(image=x)['image']
        
        if(self.train):
          #  indexing for selection by position
            y=self.df.iloc[index]['Class']
            return x,y
        else:
            return x
        
    def __len__(self):
        return len(self.df)

"""## Build train and validation datasets

While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
"""

# Split data using train_test_split
# train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets like train and validation
train, valid = train_test_split(
    data_df, 
    test_size=0.2, 
    #random state is a seed value
    random_state=42,
    # splitting a data set so that each split
    stratify=data_df.Class.values
)

# reset index on both dataframes
# sets a list of integer ranging from 0 to length of data as index.
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
# set the target value for train
train_targets = train.Class.values

# targets for validation
valid_targets = valid.Class.values
# invoke function for train
train_dataset=Milano_MRI_Dataset(
    # define dataset train
    df=train,
    imfolder=BASE_DIR,
    train=True,
    # returns a self-produced dataframe with transformed values. But none returns no values
    transforms=None
)
# invoke function for validation
valid_dataset=Milano_MRI_Dataset(
    df=valid,
    imfolder=BASE_DIR,
    train=True,
    # returns a self-produced dataframe with transformed values. But none returns no values
    transforms=None
)
# perform the dataloader
# Combines a dataset and a sampler, and provides an iterable over the given train dataset
train_loader = DataLoader(
    train_dataset,
    # defines the number of samples that will be propagated through the network.
    batch_size=15,
    #num_workers=4,
    # the data is only shuffled when the DataLoader is called as a generator or as iterator
    shuffle=True,
)
# Combines a dataset and a sampler, and provides an iterable over the given validation dataset
valid_loader = DataLoader(
    valid_dataset,
    # defines the number of samples that will be propagated through the network.
    batch_size=15,
    #num_workers=4,
    # the data is only shuffled when the DataLoader is called as a generator or as iterator
    shuffle=False,
)

"""## 3D convolutional neural network

3D convolutions applies a 3 dimentional filter to the dataset and the filter moves 3-direction (x, y, z) to calcuate the low level feature representations. Their output shape is a 3 dimentional volume space such as cube or cuboid. They are helpful in event detection in videos, 3D medical images etc.

The learning rate, or the aggressiveness with which the optimizer (in our case, the Adam optimizer) will attempt to improve once the gradient is known, is set to 0.001.

We obviously have 2 classes (0 for HC, 1 for FEP), so no_classes is 2.

Twenty percent or 0.2 of the training data is used as validation data, so this defines our validation_split.

Here used the Keras Sequential API with Conv3D, MaxPooling3D, Flatten and Dense layers.

"""

# import packages
#  torch used to building deep learning
import torch
import torch.nn as nn
import math
from functools import partial
from torch.autograd import Variable

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import datetime
import copy

# define the 3D cnn function
class Simple3DCNN(nn.Module):

    def __init__(self, num_classes):

        # avoid referring to the base class explicitly and perform multiple inheritance
        super(Simple3DCNN, self).__init__()
        # set the CNN layer 1
        self.conv_layer1 = self._make_conv_layer(1, 32)
        # set the CNN layer 2
        self.conv_layer2 = self._make_conv_layer(32, 64)
        # set the CNN layer 3
        self.conv_layer3 = self._make_conv_layer(64, 128)
        # set the CNN layer 4
        self.conv_layer4 = self._make_conv_layer(128, 256)
        # set the CNN layer 5
        # creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
        self.conv_layer5=nn.Conv3d(256, 2048, kernel_size=(4, 4, 5), padding=0)
        #Applies a linear transformation to the incoming data
        self.fc5 = nn.Linear(2048, 512)
        self.relu = nn.LeakyReLU()
        # Batch Normalization over the 3D a mini-batch of 1D inputs with optional additional channel dimension
        self.batch0=nn.BatchNorm1d(512)
        #  implemented per-layer in a neural network
        self.drop1=nn.Dropout(p=0.2)        
        self.fc6 = nn.Linear(512, 256)
        #  activation function in CNN and best alpha for it
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(256)
        self.drop2=nn.Dropout(p=0.2)
        self.fc7 = nn.Linear(256, 128)
        self.relu = nn.LeakyReLU()
        self.batch2=nn.BatchNorm1d(128)
         #  implemented per-layer in a neural network
        self.drop=nn.Dropout(p=0.25)
        self.fc8 = nn.Linear(128, num_classes)
# convontual layer
    def _make_conv_layer(self, in_c, out_c):
      # Each Linear Module computes output from input using a linear function, and holds internal Tensors for its weight and bias
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
        nn.LeakyReLU(),
        #nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        #nn.LeakyReLU(),
        nn.BatchNorm3d(out_c),
        # taking the maximum value over an input window of size defined by pool_size for each channel of the input
        nn.MaxPool3d(kernel_size=2, stride=2),
        )
        # return the value
        return conv_layer
# computes output Tensors from input Tensors
    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc5(x)
        x = self.relu(x)
        #print(x.size())
        x = self.batch0(x)
        x = self.drop1(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop2(x)
        x = self.fc7(x)        
        x = self.relu(x)
        x = self.batch2(x)

        x = self.drop(x)
        x = self.fc8(x)

        return x#,x1

# perform the 3Dcnn 
# invoke function
model = Simple3DCNN(2)

# perform iteration 
# iterator to get the first iteration. Running next() again will get the second item of the iterator.
tmp=next(iter(train_loader))
out=model(tmp[0])

"""## Defining the training function"""

# define the train function
# makes a model object using the specified CNN algorithm
def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs, device):
    # returns the time as a floating point number expressed in seconds since the epoch
    since = time.time()
    # declare the variable
    losses_train = []
    acc_train = []
    losses_val = []
    acc_val = []
    # makes the mutable OrderedDict instance not to mutate state_dict as it goes.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
# set the epoch
# a for loop that iterates over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # train and validation
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            #load input and labels
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels=labels.to(device)

                # Zero out the grads
                ## a clean up step for PyTorch
                optimizer.zero_grad()

                # Forward
                # Track history in train mode
                with torch.set_grad_enabled(phase == 'train'):
                  # moves the model to the device
                    model=model.to(device)
                    outputs = model(inputs)
                    # Returns the maximum value of all elements in the input tensor
                    _, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels.type(torch.LongTensor).unsqueeze(1).to(device))
                    #  compute the total loss
                    loss = criterion(outputs, labels.type(torch.LongTensor).to(device))

                    if phase == 'train':
                      # compute updates for each parameter
                        loss.backward()
                         # make the updates for each parameter
                        optimizer.step()
                # Statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # perform epoch loss and accuracy
            epoch_loss = running_loss/len(datasets[phase])
            epoch_acc = running_corrects.double()/len(datasets[phase])
            if phase == 'train':
              # corresponding learning rate is adjusted once according to the policy. The step() is placed in mini-batch, then step_size means that after so many iterations, the learning rate changes once.
                scheduler.step()
                  # save the current training loss information
                losses_train.append(epoch_loss)
                  # save the current training accuracy information
                acc_train.append(epoch_acc)       
                # print loss and accuracy     
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'valid':
              # save the current validation loss information
                losses_val.append(epoch_loss)
                # save the current validation accuracy information
                acc_val.append(epoch_acc) 
                # compare the training accuracy and validation accuracy
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # makes the mutable OrderedDict instance not to mutate state_dict as it goes.
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    # print the training time
    time_elapsed = time.time()-since
    # compute the elapsed time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print the best accuracy
    print('Best val Acc: {:4f}'.format(best_acc))
    # load the model
    model.load_state_dict(best_model_wts)
    # Return the model,losses_train, acc_train, losses_val, acc_val
    return model,losses_train, acc_train, losses_val, acc_val

"""## Training

In the model training, we can observe that the accuracy on validation set is fluctuating which suggests that the network can be improved further. Let's predict and measure the accuracy of current model

We use a epoch of 100 samples. This means that one hundred samples are fed forward through the network each time, generating predictions, computing loss, and optimization. The higher the batch size, the higher the efficiency with which the improvement gradient can be computed, but the more memory is required.
"""

# coutn the class
class_sample_count = np.array([len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)])
weight = 1. / class_sample_count
#The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. The returned tensor is not resizable.
class_weight=torch.from_numpy(weight)
class_weight=class_weight.max()/class_weight
class_weight=class_weight/class_weight.max()
# not to overwrite the imported module
class_weight=class_weight.type(torch.FloatTensor)
print(class_weight)

# to move a tensor to a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train and validation
datasets={'train':train_dataset,'valid':valid_dataset}
dataloaders={'train':train_loader,'valid':valid_loader}
# implements Adam algorithm with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.001)
# Decays the learning rate of each parameter group by gamma every step_size epochs. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# combines LogSoftmax and NLLLoss in one single class. It is useful when training a classification problem with C classes.
criterion=nn.CrossEntropyLoss()
#criterion=nn.BCEWithLogitsLoss()
# set the epoch value
num_epochs=100
# perform the model
trained_model,losses_train, acc_train, losses_val, acc_val=train_model(datasets,dataloaders,model,criterion,optimizer,scheduler,num_epochs,device)

"""## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted. Since the validation set is class-balanced, accuracy provides an unbiased representation of the model's performance.
"""

#plotting the loss for training
plt.plot(losses_train)
# set title
plt.title('Loss vs Epochs Train')
# set x label
plt.xlabel('Epochs')
# set y label
plt.ylabel('loss')

#printing the accuracy for training
plt.plot(acc_train)
# set title
plt.title('Accuracy vs Epochs Train')
# set x label
plt.xlabel('Accuracy')
# set y label
plt.ylabel('Epochs')

#plotting the loss for validation
plt.plot(losses_val)
# set title
plt.title('Loss vs Epochs Validate')
# set x label
plt.xlabel('Epochs')
# set y label
plt.ylabel('loss')

#printing the accuracy for validation
plt.plot(acc_val)
# set title
plt.title('Accuracy vs Epochs Validate')
# set x label
plt.xlabel('Accuracy')
# set y label
plt.ylabel('Epochs')

"""## Testing

Here confusion matrix has been used for summarizing the performance of a 3D CNN classification model. Classification accuracy alone can be misleading is some cases. So along with that the confusion matrix also developed to find the models performance. 
"""

# evaluation
# switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time
trained_model.eval()
y_pred_list = []
#  Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor. backward()
with torch.no_grad():
  for inputs, labels in valid_loader:
    # does not change inputs, but rather returns a copy of inputs that resides on device
    inputs = inputs.to(device)
    outputs = trained_model(inputs)
    # prediction
    _, preds = torch.max(outputs, 1)
    arr_pred = preds.tolist()
    for cls in arr_pred:
      # add the cls details to y_pred_list
      y_pred_list.append(cls)
      #  convert input to an array.
y_pred_list = np.asarray(y_pred_list)

# confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
#  List of labels to index the matrix
confusion_matrix = confusion_matrix(valid_targets, y_pred_list)
# print result
confusion_matrix

# confusion matrix visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
# Convert the dataframe
confusion_matrix = pd.DataFrame(confusion_matrix, range(2),range(2))
# set figure size
plt.figure(figsize = (10,10))
# draw the heat map
sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 12}) # font size
# show the heat map
plt.show()

"""A classification report is used to show the precision, recall, F1 Score, and support of your trained classification model.

Here the precision is defined as the ratio of true positives to the sum of true and false positives.

Recall is defined as the ratio of true positives to the sum of true positives and false negatives.

The F1 is the weighted harmonic mean of precision and recall. The closer the value of the F1 score is to 1.0, the better the expected performance of the model is.

Support is the number of actual occurrences of the class in the dataset. It doesn’t vary between models, it just diagnoses the performance evaluation process.
"""

# classification report
print(classification_report(valid_targets, y_pred_list))