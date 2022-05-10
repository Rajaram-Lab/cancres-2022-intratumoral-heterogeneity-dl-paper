"""
MILCore contains all core functions used to train the MIL model that is used for 
slide level classification. It will be called to access specific functions for 
data generators and model architectures.

This code was adapted from the following repositories: 
https://github.com/utayao/Atten_Deep_MIL
https://github.com/AMLab-Amsterdam/AttentionDeepMIL

We thank the authors of these reposistories for their work.

 Copyright (C) 2021, Rajaram Lab - UTSouthwestern 
    
    This file is part of cancres-2022-intratumoral-heterogeneity-dl-paper.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with cancres-2022-intratumoral-heterogeneity-dl-paper.  If not, see <http://www.gnu.org/licenses/>.
    
    Paul Acosta, Vipul Jarmale 2022
"""


import os,  sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0, ROOT_DIR)

import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import  initializers, regularizers

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import  Layer, Dropout, Conv2D, Flatten, multiply

from tensorflow.keras.applications import  VGG19


# %%

class MILSlideGen(Sequence):
    def __init__(self, data, labels, sampleNumbers,
                numberOfClasses, batch_size=64, shuffle=True, 
                preproc_fn=lambda x: np.float32(x)/255,
                categoricalLabels=True, augmentations=None):
        
        self.data = data
        self.labels = labels
        self.sampleNumbers = sampleNumbers
        self.numberOfClasses = numberOfClasses
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preproc = preproc_fn
        self.categoricalLabels = categoricalLabels
        self.augmentations = augmentations
        
        self.numberOfPatches = data.shape[0]
        self.numberOfBatches = int(np.floor(self.numberOfPatches / self.batch_size))
        self.uniqueSampleNumbers = np.unique(self.sampleNumbers) # save to iterate thru unique sample numbers
        
        self.sampleIdx={} # a dictionary to save all patch indices contributed by a sample
        for sampleNumber in self.uniqueSampleNumbers:
            self.sampleIdx[sampleNumber] = np.where(self.sampleNumbers == sampleNumber)[0]
            labelsForSample = self.labels[self.sampleIdx[sampleNumber]] # labels found for current sample
            if len(np.unique(labelsForSample)) != 1:
                raise Exception("Multiple labels found for current sample")
        
        self.pSample = np.array([len(self.sampleIdx[sampleNumber]) for sampleNumber in self.uniqueSampleNumbers])/self.numberOfPatches
        self.on_epoch_end()
    
    
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return self.numberOfBatches
    
    
    def on_epoch_end(self):
        self.indexList = [] # flush out the self.indexList at the end of epoch
        for _ in range(self.numberOfBatches):
            # choose a sample number based on the patches contributed in the data
            sampleNumber = int(np.random.choice(self.uniqueSampleNumbers, p=self.pSample))
            # sample indices for current sample number and append to index list
            self.indexList.append(np.random.choice(self.sampleIdx[sampleNumber], size=self.batch_size))
    
    
    def __getitem__(self, index):
        """ Generate one batch of data """
        
        # access the specified element (containing indices) from list of index-arrays
        indexes = self.indexList[index]
        
        # first, init a list of ones as long as the batch size
        # if it's a non-focal case, all labels are 1 => np.min becomes 1. Labels therefore are all ones.
        # in case of focal cases, all labels are 0 => np.min becomes 0. Labels are array becomes all zeroes.
        labels = np.ones(len(indexes)) * np.min(self.labels[indexes])
        # change labels to required format
        outLabels = to_categorical(labels, num_classes=self.numberOfClasses) if self.categoricalLabels else labels
        
        # initialize a numpy array of shape (batchSize, width, height, numChannels)
        # perform augmentation on each patch in the batch and save in the dataOut array
        dataOut = np.zeros((len(indexes), self.data.shape[1], self.data.shape[2], self.data.shape[3]), dtype=np.float32)
        for i, idx in enumerate(indexes):
            currPatch = np.squeeze(self.data[idx, :, :,:])
            if self.augmentations is not None:
                currPatch = self.augmentations.augment_image(currPatch)
            dataOut[i, :, :, :] = self.preproc(currPatch)
        return dataOut, outLabels

# %% Custom Layers

class Last_Sigmoid(Layer):
    """
    Attention Activation

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.   !!!!!!!!!!!!!!!!!!!

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
   
    def __init__(self, output_dim=1, kernel_initializer='glorot_uniform', bias_initializer='zeros',name='FC1_sigmoid',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, **kwargs):

        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = int(input_shape[1])

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = int(self.output_dim)
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Normalize_Layer(Layer):

    def __init__(self,  **kwargs):


        super(Normalize_Layer, self).__init__(**kwargs)



    def call(self, x):
        mag=1/(K.sum(x,axis=0)+1E-6)
        out = x*mag
        # compute instance-level score

        return out

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):
        config = {
        }
        base_config = super(Normalize_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# %% Losses and Metrics
def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    acc = K.mean(K.equal(y_true, K.round(y_pred)))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    loss : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def weighted_bag_loss(weightsList):
    def lossFunc(true, pred):
        y_true = K.cast(K.mean(true, axis=0, keepdims=False), dtype='int32')
        loss = bag_loss(true,pred)
        loss = loss*K.gather(weightsList,y_true)
        return loss
    
    return lossFunc


def MilNetwork(input_dim, params, class_weights,numberOfClasses=2,useMulGpu=False):
    
    if numberOfClasses != 2:
        raise SystemExit('sigmoid activation only supports two output classes')
    
    lr = params['MIL']['lr']
    weight_decay = params['MIL']['decay']
    momentum = params['MIL']['momentum']
    model = VGG19(weights = "imagenet", include_top=False, input_shape = input_dim)

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False
    
    #Adding custom Layers
    x = model.output
    x = Conv2D(1024,kernel_size=(7,7),kernel_regularizer=l2(weight_decay),activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Conv2D(1024,kernel_size=(1,1),kernel_regularizer=l2(weight_decay),activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    y = model.output
    y = Conv2D(1024,kernel_size=(7,7),activation='relu')(y)

    y = Conv2D(1,kernel_size=(1,1),activation='sigmoid')(y)
    
    y = Flatten()(y)
    
    alpha = Normalize_Layer()(y)
    x_mul = multiply([alpha,x])
    
   
    out = Last_Sigmoid(output_dim=1)(x_mul)
    
    model = Model(inputs= model.input, outputs = out)
    
    model.compile(optimizer=SGD(lr=lr,momentum=momentum), loss=weighted_bag_loss(class_weights), metrics=[bag_accuracy])

    return model
