#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:25:12 2020

@author: gwenda
"""


# begin by importing our dependencies.
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import spike
import testspike

#import ipdb

# set our seed and other configurations for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# set the batch size, the number of training epochs, and the learning rate
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_NEURONS = [128, 64]

IN_SHAPE = 784
OUT_SHAPE = 10

DURATION = 16

SPIKE_LEARNING_RATE = 1e-4


# use gpu if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# classifier class with fully connected layers for both encoder and decoder
class CF(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
    creates four layers, two for the encoder and two for the decoder
    '''
    def __init__(self, **kwargs):
        super().__init__()

        self.input_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                     out_features=HIDDEN_NEURONS[0])
        
        self.hidden_layer = nn.Linear(in_features=HIDDEN_NEURONS[0], 
                                      out_features=HIDDEN_NEURONS[1])
        
        self.output_layer = nn.Linear(in_features=HIDDEN_NEURONS[1], 
                                      out_features=kwargs["output_shape"])
        
        self.layers = [self.input_layer, self.hidden_layer, self.output_layer]
        


        
    '''
    a method updated from the super class, runs input through all the layers
    '''
    def forward(self, images):
        self.activations = []
        softmax = nn.LogSoftmax(dim=1)
        lrelu = nn.LeakyReLU()
        features = torch.relu(self.layers[0](images))
        self.activations.append(features)
        features = torch.relu(self.layers[1](features))
        self.activations.append(features)
        features = lrelu(self.layers[2](features))
        self.activations.append(features)
        
        output = softmax(features)
        return output

    
# classifier spiking network
class CF_spikes(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
           of pretrained_model ~ use to recreate the weights exactly
    creates four layers, two for the encoder and two for the decoder, also 
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = []
        
        # make the layers
        self.encoder_input_layer = spike.Encoder(num_features=kwargs["input_shape"])

        self.input_layer = spike.Linear(in_features=kwargs["input_shape"], 
                                              out_features=HIDDEN_NEURONS[0])
        
        self.hidden_layer = spike.Linear(in_features=HIDDEN_NEURONS[0], 
                                              out_features=HIDDEN_NEURONS[1])
        
        self.output_layer = spike.Linear(in_features=HIDDEN_NEURONS[1], 
                                                out_features=kwargs["output_shape"])

        
        # now append all the layers to a list.
        self.layers.append(self.encoder_input_layer)
        self.layers.append(self.input_layer)
        self.layers.append(self.hidden_layer)
        self.layers.append(self.output_layer)



    def _save_memory(self, in_features):
        """internal method. This executes the network in continuous mode for a single input.
        Activations are recorded so weigths can be normalized for spking execution.
        """
        # Execute the network on this input, and have eachlayer record activations.
        softmax = nn.LogSoftmax(dim=1)
        features = torch.relu(self.input_layer.save_memory(in_features))
        features = torch.relu(self.hidden_layer.save_memory(features))
        softmax(self.output_layer.save_memory((features)))


    def _create_bins(self, num_bins):
        '''
        uses the the memory in order to create a list of bins for the spiking
        network to use for discretization purposes
        '''
        # Skip spike encoder layer because this is executing in continuous mode.
        for layer in self.layers[1:]:
            layer.create_bins(num_bins)
    
        
    # This can still be simplified and handled mostly by the layers.
    # TODO: Combine activation range computation (_save_memory) (bins)
    # with normalizing weights (_set_up_weights) and put in layers.
    def translate(self, train_loader, duration=16):
        """ Use loaded continuous weights and convert to spiking network weights and state.
        test_loader: (torch.utils.data.DataLoader) source images to inspect activation ranges.
        duration: number of time steps used to count spikes to compute firing rate.
        """
        with torch.no_grad():
            for batch_features in train_loader:
                batch_features = batch_features[0]
                test_examples = batch_features.view(-1, IN_SHAPE)
                #self._save_memory(test_examples.to(DEVICE))
                self._save_memory(test_examples)
                break
        
        self._create_bins(duration)

        # Adjust weights to normalize the firing rates.
        for layer in self.layers:
            layer.translate_weights()

        
    def reset(self):
        """
        resets the potentials to zero so that a new image can be processed
        """
        for layer in self.layers:
            layer.reset()

            
    '''
    a method updated from the super class, runs input through all the layers
    '''
    def forward(self, features):
        # loop through every image in the batch
        # reset the potentials to zero
        self.reset()

        # loop through every layer until we've finished processing the spikes
        for i in range(DURATION):
            x = features
            for layer in self.layers:
                x = layer.process(x)

        # convert the output spikes to continuous voltages
        output_layer = self.layers[-1]
        output = output_layer.get_activation()
            
        return output

    
    def forward_learn(self, features, layer_idx, teacher, learning_rate):
        # +1 is to skip the spike enocder
        layer = self.layers[layer_idx+1]

        # Get the truth (output activation of the teachers target layer).
        # Execute the teacher using the supplied input batch.
        teacher(features)
        # Get the activations for the target layer.
        layer_truth = teacher.activations[layer_idx]
        # This should scale the activations to 0->1
        layer_truth /= layer.out_bins[-1]

        # Execute the netowrk
        self.forward(features)

        # Let the layer learn to generate the truth (instead of its current activation).
        error = layer.learn_spike_frequencies(layer_truth, learning_rate)
        return error
    
    
    def last_layer_learn(self, features, labels, learning_rate):
        # Execute the spiking network
        self.forward(features)

        layer = self.layers[-1]
        error = layer.learn_labels(labels, learning_rate)
        return error



        
def load_mnist():
    # load MNIST dataset using the torchvision package.
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", 
                                               train=True, transform=transform, 
                                               download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True)
    
    # get some test examples
    test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", 
                                              train=False, transform=transform, 
                                              download=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, 
                                              shuffle=False)
    return train_loader, test_loader



def train_classifier(model, train_loader):
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=10*LEARNING_RATE, momentum=0.9)
    
    # mean-squared error loss
    criterion = nn.NLLLoss()
    
    # train our classifier for our specified number of epochs.
    for epoch in range(EPOCHS):
        loss = 0
        for images,labels in train_loader:
            # Flatten MNIST images into a IN_SHAPE long vector
            images = images.view(images.shape[0], -1).to(DEVICE)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(images)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, labels.to(DEVICE))
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, EPOCHS, loss))




def compute_accuracy(model, test_loader):
    total_correct = 0
    total = 0
    # create the reconstructions using the model
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            pred_labels = model(images.to(DEVICE))
            for idx in range(len(pred_labels)):
                if torch.argmax(pred_labels[idx]) == labels[idx]:
                    total_correct += 1
                total += 1

    return (100.0 * total_correct/total)
# 8(5),92(9)

#[0.0625, 0.0000, 0.0625, 0.0625, 0.1875, 0.3125, 0.3125, 0.0000, 0.1875, .1250]
#[0.0625, 0.0000, 0.0625, 0.0000, 0.1875, 0.2500, 0.3125, 0.0000, 0.1875, 0.1250]


#[0.0000, 0.0625, 0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.1250, 0.0625, 0.2500]


def train_spikingnet(model, teacher, train_loader, layer_idx, learning_rate):
    """ One epoch.  Return total error.
    """
    # create the reconstructions using the model
    errors = []
    with torch.no_grad():
        for batch_features in train_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, IN_SHAPE)
            errors.append(model.forward_learn(test_examples.to(DEVICE),layer_idx,
                                              teacher, learning_rate))
    return sum(errors)/len(errors)


def train_spiking_lastlayer(model, teacher, train_loader, learning_rate):
    """ One epoch.  Return total error.
    """
    # create the reconstructions using the model
    errors = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            errors.append(model.last_layer_learn(images.to(DEVICE),labels.to(DEVICE),
                                                 learning_rate))
    return sum(errors)/len(errors)
            
        
def test_mnist_classifier(show=True):
    results = {}
    train_loader, test_loader = load_mnist()

    # check if we've already trained this up
    model = CF(input_shape=IN_SHAPE,output_shape=OUT_SHAPE).to(DEVICE)

    if os.path.isfile('test_data/mnist_classifier.pth'):
        model.load_state_dict(torch.load('test_data/mnist_classifier.pth',
                                         map_location=lambda storage, loc: storage))
    else:
        # create a model from `CF` classifier class and load it to gpu    
        # train it
        train_classifier(model, train_loader)
        # save the model
        torch.save(model.state_dict(), 'test_data/mnist_classifier.pth')

    # compute the accuracy of the original continuous network
    results['accuracy_CF'] = compute_accuracy(model, test_loader)
    print("Accuracy of OG Model:")
    print(results['accuracy_CF'])


    # create a spiking model
    spiking_model = CF_spikes(input_shape=IN_SHAPE,output_shape=OUT_SHAPE)
    # Copy in the trained weights.
    spiking_model.load_state_dict(model.state_dict())
    spiking_model.translate(train_loader, DURATION)
    spiking_model.to(DEVICE)

    # compute the accuracy of the spiking model
    results['accuracy_CF_spikes'] = compute_accuracy(spiking_model, test_loader)
    print("Accuracy of Spiking Model:")
    print(results['accuracy_CF_spikes'])

    #numpy_results = []
    

    ## Tune/train the spiking network
    error = 100000
    for layer_idx in range(3):
        print("Layer %d"%(layer_idx+1))
        learning_rate = SPIKE_LEARNING_RATE
        for epoch_idx in range(10):
            print("  Epoch %d"%(epoch_idx+1))
            tmp = train_spikingnet(spiking_model, model, test_loader,
                                   layer_idx, learning_rate)
            if tmp > error:
                learning_rate *= 0.5
            error = tmp
            print(f"    error = {error}")
            accuracy_CF_spikes_train = compute_accuracy(spiking_model, train_loader)
            print(accuracy_CF_spikes_train)
            
    print('Last Layer Train')
    learning_rate = SPIKE_LEARNING_RATE
    for epoch_idx in range(10):
        print("  Epoch %d"%(epoch_idx+1))
        tmp = train_spiking_lastlayer(spiking_model, model,
                                      test_loader, learning_rate)
        if tmp > error:
            learning_rate *= 0.5
        error = tmp
        print(f"    error = {error}")
        accuracy_CF_spikes_train = compute_accuracy(spiking_model, train_loader)
        print(accuracy_CF_spikes_train)
            
    results['accuracy_CF_spikes_train'] = accuracy_CF_spikes_train
    print(results['accuracy_CF_spikes_train'])
    
    return results



testspike.evaluate_results(test_mnist_classifier(show=False), 'mnist_classifier')

