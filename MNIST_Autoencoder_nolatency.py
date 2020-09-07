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
EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_NEURONS = 128
IN_SHAPE = 784

DURATION = 16

SPIKE_LEARNING_RATE = 1e-4


# use gpu if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# autoencoder class with fully connected layers for both encoder and decoder
class AE(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
    creates four layers, two for the encoder and two for the decoder
    '''
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                              out_features=HIDDEN_NEURONS)
        
        self.encoder_output_layer = nn.Linear(in_features=HIDDEN_NEURONS, 
                                              out_features=HIDDEN_NEURONS)
        
        self.decoder_hidden_layer = nn.Linear(in_features=HIDDEN_NEURONS, 
                                              out_features=HIDDEN_NEURONS)
        
        self.decoder_output_layer = nn.Linear(in_features=HIDDEN_NEURONS, 
                                                out_features=kwargs["input_shape"])
        
        self.layers = [self.encoder_hidden_layer, self.encoder_output_layer,
                       self.decoder_hidden_layer, self.decoder_output_layer]

        
    '''
    a method updated from the super class, runs input through all the layers
    '''
    def forward(self, images):
        self.activations = []
        features = torch.relu(self.layers[0](images))
        self.activations.append(features)
        features = torch.relu(self.layers[1](features))
        self.activations.append(features)
        features = torch.relu(self.layers[2](features))
        self.activations.append(features)
        features = torch.relu(self.layers[3](features))
        self.activations.append(features)
        
        output = features
        return output




    
# autoencoder spiking network
class AE_spikes(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
           of pretrained_model ~ use to recreate the weights exactly
    creates four layers, two for the encoder and two for the decoder, also 
    '''
    def __init__(self, input_shape):
        super().__init__()
        self.layers = []
        
        # make the layers
        self.encoder_input_layer = spike.Encoder(num_features=input_shape)

        self.encoder_hidden_layer = spike.Linear(in_features=input_shape, 
                                                 out_features=HIDDEN_NEURONS)
        
        self.encoder_output_layer = spike.Linear(in_features=HIDDEN_NEURONS, 
                                                 out_features=HIDDEN_NEURONS)
        
        self.decoder_hidden_layer = spike.Linear(in_features=HIDDEN_NEURONS, 
                                                 out_features=HIDDEN_NEURONS)
        
        self.decoder_output_layer = spike.Linear(in_features=HIDDEN_NEURONS, 
                                                 out_features=input_shape)  
        
        
        # now append all the layers to a list.
        self.layers.append(self.encoder_input_layer)
        self.layers.append(self.encoder_hidden_layer)
        self.layers.append(self.encoder_output_layer)
        self.layers.append(self.decoder_hidden_layer)
        self.layers.append(self.decoder_output_layer)


    def _save_memory(self, in_features):
        """internal method. This executes the network in continuous mode for a single input.
        Activations are recorded so weigths can be normalized for spking execution.
        """
        # Execute the network on this input, and have eachlayer record activations.
        features = torch.relu(self.layers[1].save_memory(in_features))
        features = torch.relu(self.layers[2].save_memory(features))
        features = torch.relu(self.layers[3].save_memory(features))
        torch.relu(self.layers[4].save_memory(features))


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
    def translate(self, test_loader, duration=16):
        """ Use loaded continuous weights and convert to spiking network weights and state.
        test_loaded: (torch.utils.data.DataLoader) source images to inspect activation ranges.
        duration: number of time steps used to count spikes to compute firing rate.
        """
        with torch.no_grad():
            for batch_features in test_loader:
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
    
    
    def last_layer_learn(self, features, learning_rate):
        # Execute the spiking network
        self.forward(features)

        layer = self.layers[-1]
        error = layer.learn_spike_frequencies(features/layer.out_bins[-1], learning_rate)
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



def train_autoencoder(model, train_loader):
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # mean-squared error loss
    criterion = nn.MSELoss()
    
    # train our autoencoder for our specified number of epochs.
    for epoch in range(EPOCHS):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, IN_SHAPE] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, IN_SHAPE).to(DEVICE)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
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


def reconstruct_test_images(model, test_loader, show=True):
    test_examples = None

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, IN_SHAPE)
            reconstruction = model(test_examples.to(DEVICE))
            break
    
    # reconstruct some test images using trained autoencoder
    results = []
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            out_img = reconstruction[index].cpu().numpy().reshape(28, 28)
            if show:
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.imshow(test_examples[index].numpy().reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                results.append(out_img)
                plt.imshow(out_img)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if show:
            plt.show()

    return np.array(results)


def compute_mMSE(model, test_loader):
    test_examples = None
    mMSE = []
    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, IN_SHAPE)
            reconstruction = model(test_examples.to(DEVICE))
            # TODO: make it so you don't loop over the images
            for index in range(len(reconstruction)):
                # original
                y = test_examples[index].numpy()
        
                # reconstruction
                y_pred = reconstruction[index].cpu().numpy()
                
                mse = np.mean((y - y_pred)**2)
                
                mMSE.append(mse)
    

        return np.mean(np.array(mMSE))


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
            errors.append(model.last_layer_learn(images.to(DEVICE),learning_rate))
    return sum(errors)/len(errors)


# save out the weights for the first layer, they look vaguely like parts of
# the handwritten numbers
def save_weight_images(weights):
    for idx in len(HIDDEN_NEURONS):
        np.save('neuron_{}'.format(idx+1),weights[0][idx].reshape(28, 28))


# get the bins from the model using the test dataset
#def compute_bins(model, test_loader, bits):
#    test_examples = None
#
#    with torch.no_grad():
#        for batch_features in test_loader:
#            batch_features = batch_features[0]
#            test_examples = batch_features.view(-1, IN_SHAPE)
#            model.save_memory(test_examples.to(DEVICE))
#            break
#        
#    model.create_bins(bits)
    
        
def test_mnist_autoencoder(show=True):
    results = {}
    train_loader, test_loader = load_mnist()

    # check if we've already trained this up
    model = AE(input_shape=784).to(DEVICE)

    if os.path.isfile('test_data/mnist_autoencoder.pth'):
        model.load_state_dict(torch.load('test_data/mnist_autoencoder.pth',
                                         map_location=lambda storage, loc: storage))
    else:
        # create a model from `AE` autoencoder class and load it to gpu    
        # train it
        train_autoencoder(model, train_loader)
        # save the model
        torch.save(model.state_dict(), 'test_data/mnist_autoencoder.pth')

    # compute the mse of the original continuous network
    results['mMSE_AE'] = compute_mMSE(model, test_loader)
    print("Mean Squared Error of OG Model:")
    print(results['mMSE_AE'])


    # create a spiking model
    spiking_model = AE_spikes(input_shape=IN_SHAPE)
    # Copy in the trained weights.
    spiking_model.load_state_dict(model.state_dict())
    spiking_model.translate(test_loader, DURATION)
    spiking_model.to(DEVICE)

    # compute the mse of the spiking model
    results['mMSE_AE_spikes'] = compute_mMSE(spiking_model, test_loader)
    print("Mean Squared Error of Spiking Model:")
    print(results['mMSE_AE_spikes'])

    #numpy_results = []
    
    results['continuous_reconstruction'] = reconstruct_test_images(model, test_loader, show)
    results['spiking_reconstruction'] = reconstruct_test_images(spiking_model, test_loader, show)

    
    ## Tune/train the spiking network
    error = 100000

    for layer_idx in range(4):
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
            mMSE_AE_spikes_train = compute_mMSE(spiking_model, test_loader)
            print(mMSE_AE_spikes_train)
            
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
        mMSE_AE_spikes_train = compute_mMSE(spiking_model, test_loader)
        print(mMSE_AE_spikes_train)
        
            
    results['mMSE_AE_spikes_train'] = mMSE_AE_spikes_train
    print(results['mMSE_AE_spikes_train'])
    results['tuned_reconstruction'] = reconstruct_test_images(spiking_model, test_loader, show)
    
    return results



testspike.evaluate_results(test_mnist_autoencoder(show=False), 'mnist_autoencoder')




"""
mMSEs_spikes = []
bits_list = []

for bit_depth in range(2,64):
    bits = bit_depth
    bins = compute_bins(model, test_loader, bits)

    # create a spiking model
    spiking_model = AE_spikes(bits=bits,bins=bins,input_shape=IN_SHAPE,
                              pretrained_model=model).to(DEVICE)
    
    
    # compute the mse of the spiking model
    mMSEs_spikes.append(compute_mMSE(spiking_model, test_loader))
    bits_list.append(bits)
    
fig, ax = plt.subplots()
ax.plot(bits_list, mMSEs_spikes)
ax.hlines(mMSE_AE,2,63)
ax.set(xlabel='# Discrete Bins', ylabel='Average MSE',
       title='Accuracy vs Latency')


plt.show()

"""
