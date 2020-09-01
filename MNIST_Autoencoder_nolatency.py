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
import test

import ipdb

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

lr = 1e-4


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
    def forward(self, features):
        activation = self.encoder_hidden_layer(features) # performs matmul of input and weights
        activation = torch.relu(activation) # only output vals >= 0
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation_decoder = self.decoder_hidden_layer(code)
        activation_decoder = torch.relu(activation_decoder)
        reconstructed = self.decoder_output_layer(activation_decoder)
        reconstructed = torch.relu(reconstructed)
        
        return reconstructed




    
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
        activation = self.encoder_hidden_layer.save_memory(in_features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer.save_memory(activation)
        code = torch.relu(code)
        activation_decoder = self.decoder_hidden_layer.save_memory(code)
        activation_decoder = torch.relu(activation_decoder)
        reconstructed = self.decoder_output_layer.save_memory(activation_decoder)
        reconstructed = torch.relu(reconstructed)


    def _create_bins(self, num_bins):
        '''
        uses the the memory in order to create a list of bins for the spiking
        network to use for discretization purposes
        '''
        self.encoder_hidden_layer.create_bins(num_bins)
        self.encoder_output_layer.create_bins(num_bins)
        self.decoder_hidden_layer.create_bins(num_bins)
        self.decoder_output_layer.create_bins(num_bins)
    
        
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


    def copy(self, in_model):
        """ Convert might be a better name.
        Custom for the autoencoder, but will be made general in the future.
        """
        # source net does not have an encoder_input_layer.
        self.encoder_hidden_layer.copy(in_model.encoder_hidden_layer)
        self.encoder_output_layer.copy(in_model.encoder_output_layer)
        self.decoder_hidden_layer.copy(in_model.decoder_hidden_layer)
        self.decoder_output_layer.copy(in_model.decoder_output_layer)

        self.set_up_weights()

        
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
        reconstructions = []
        # loop through every image in the batch
        for feature in features:
            # reset the potentials to zero
            self.reset()
            # divide input into 15
            # TODO: Try to get rid of this call.
            input_activation = self.encoder_hidden_layer.discretize(feature)/float(DURATION)

            output_spikes = torch.zeros(IN_SHAPE).to(DEVICE)
            
            # loop through every layer until we've finished processing the spikes
            for i in range(DURATION):
                x = input_activation
                for layer in self.layers:
                    x = layer.process(x)
                output_spikes += x
                
            # convert the output spikes to voltages
            reconstructed = self.decoder_output_layer.reconstruct(output_spikes)
            # add the reconstructed image to the list
            reconstructions.append(reconstructed)
            
        return reconstructions
    
    def forward_learn(self, features, layer_idx, teacher):
        reconstructions = []

        # loop through every image in the batch
        for feature in features:
            # reset the potentials to zero
            self.reset()
            # divide input into 16 (duration)
            input_activation = self.encoder_hidden_layer.discretize(feature)/float(DURATION)

            # loop through every layer until we've finished processing the spikes
            for i in range(DURATION):
                x = input_activation
                for layer in self.layers:
                    x = layer.process(x)

            # spike count of output layer.
            output_spikes = self.layers[-1].get_spike_counts()

            # get the outputs from the original network for our truth
            og_layers = []
            og_layers.append(torch.relu(teacher.encoder_hidden_layer(feature)))
            og_layers.append(torch.relu(teacher.encoder_output_layer(og_layers[0])))
            og_layers.append(torch.relu(teacher.decoder_hidden_layer(og_layers[1])))
            og_layers.append(torch.relu(teacher.decoder_output_layer(og_layers[2])))

            # change to frequencies by using max bin
            # Skip the spike enocder layer.  TODO: See if we can generalize.
            og_layers[0] /= self.layers[1].out_bins[-1]
            og_layers[1] /= self.layers[2].out_bins[-1]
            og_layers[2] /= self.layers[3].out_bins[-1]
            og_layers[3] /= self.layers[4].out_bins[-1]
            
            # now we have to use the spike frequencies and og layer outputs to learn            
            if layer_idx == 0:
                spike_frequencies = self.layers[1].get_spike_frequencies()
                self.encoder_hidden_layer.weight += ((input_activation * DURATION) *\
                    torch.reshape((og_layers[0]-spike_frequencies),[128,1])) * lr
                    
                self.encoder_hidden_layer.weight += (torch.ones(input_activation.shape).to(DEVICE) *\
                    torch.reshape((og_layers[0]-spike_frequencies),[128,1])) * lr
                    
            else:
                in_frequencies = self.layers[layer_idx].get_spike_frequencies()
                out_frequencies = self.layers[layer_idx+1].get_spike_frequencies()
                # +1 is to skip the spike enocder
                self.layers[layer_idx+1].weight += ((in_frequencies * DURATION) *\
                    torch.reshape((og_layers[layer_idx]-out_frequencies),\
                                  [len(og_layers[layer_idx]),1])) * lr
                    
                self.layers[layer_idx+1].weight += (torch.ones(in_frequencies.shape).to(DEVICE) *\
                    torch.reshape((og_layers[layer_idx]-out_frequencies),\
                                  [len(og_layers[layer_idx]),1])) * lr
            
            # convert the output spikes to voltages
            reconstructed = self.layers[4].reconstruct(output_spikes)
            # add the reconstructed image to the list
            reconstructions.append(reconstructed)
            
        return reconstructions


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

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, IN_SHAPE)
            reconstruction = model(test_examples.to(DEVICE))
            break
    
    
    # reconstruct some test images using trained autoencoder
    with torch.no_grad():
        number = 100
        mMSE = []
        for index in range(number):
            # original
            y = test_examples[index].numpy()
    
            # reconstruction
            y_pred = reconstruction[index].cpu().numpy()
            
            mse = np.mean((y - y_pred)**2)
            
            mMSE.append(mse)
        
        # take the average of all those MSEs to return
        return np.mean(np.array(mMSE))


def train_spikingnet(model, teacher, train_loader, layer_idx):
    test_examples = None

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in train_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, IN_SHAPE)
            model.forward_learn(test_examples.to(DEVICE),layer_idx, teacher)
            break


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
    for i in range(5):
        print("Epoch %d"%(i+1))
        for layer_idx in range(4):
            print("Layer %d"%(layer_idx+1))
            train_spikingnet(spiking_model, model, train_loader, layer_idx)
            mMSE_AE_spikes_train = compute_mMSE(spiking_model, test_loader)
            print(mMSE_AE_spikes_train)
    results['mMSE_AE_spikes_train'] = mMSE_AE_spikes_train
    print(results['mMSE_AE_spikes_train'])
    results['tuned_reconstruction'] = reconstruct_test_images(spiking_model, test_loader, show)
    
    return results



test.evaluate_results(test_mnist_autoencoder(show=False), 'mnist_autoencoder')



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
