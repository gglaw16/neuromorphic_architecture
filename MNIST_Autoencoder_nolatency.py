#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:25:12 2020

@author: gwenda
"""


# begin by importing our dependencies.
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import ipdb

# set our seed and other configurations for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# set the batch size, the number of training epochs, and the learning rate
batch_size = 512
epochs = 20
learning_rate = 1e-3
hidden_neurons = 128
in_shape = 784

lr = 1e-4


# use gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# spiking linear layer
class SLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features)        
        # There is duplication of output/inputs bwtween layers, but
        # this allows mamangent of memeory and bins in layers.
        # An alternative is to create a spike encoding layer to manage input bins.
        self.in_memory = []
        self.in_bins = []
        self.out_memory = []        
        self.out_bins = []        

    def save_memory(self, x):
        """ This is an alternative execute function that also records input and output
        activation functions.  This is required to chose a discritization for spking.
        We may only have to save the maximum.  I think np.histogram_bin_edges(...)
        only uses max and min.  However, ouliers may cause trouble.
        """
        self.in_memory.append(x.cpu().numpy())
        x = self(x)
        tmp = torch.relu(x)
        self.out_memory.append(tmp.cpu().numpy())
        return x
        
    def create_bins(self, bits):
        self.bits = bits
        self.in_bins = np.histogram_bin_edges(self.in_memory, bins=bits)
        self.out_bins = np.histogram_bin_edges(self.out_memory, bins=bits)

    def copy(self, source_layer):
        # Copy weights and biases
        self.load_state_dict(source_layer.state_dict())
        # Copy bins
        self.in_bins = np.copy(source_layer.in_bins)
        self.out_bins = np.copy(source_layer.out_bins)

    # TODO: Try to get rid of this method.
    def discretize(self, values):
        """
        uses the bins input in the init to change input values to a discrete number
        of spikes, uses idx to figure out which list of bins to use
        """
        spikes = np.digitize(values.cpu(), self.in_bins)-1
        return torch.from_numpy(spikes).to(device)

    def reconstruct(self, output_spikes):
        return output_spikes*(self.out_bins[1]-self.out_bins[0])

        
        
# autoencoder class with fully connected layers for both encoder and decoder
class AE(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
    creates four layers, two for the encoder and two for the decoder
    '''
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_hidden_layer = SLinear(in_features=kwargs["input_shape"], 
                                              out_features=hidden_neurons)
        
        self.encoder_output_layer = SLinear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_hidden_layer = SLinear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_output_layer = SLinear(in_features=hidden_neurons, 
                                              out_features=kwargs["input_shape"])
        

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
    
    '''
    this saves to memory all the possible values for each layer's output
    basically does the same thing as forward otherwise
    '''
    def save_memory(self, features):
        activation = self.encoder_hidden_layer.save_memory(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer.save_memory(activation)
        code = torch.relu(code)
        activation_decoder = self.decoder_hidden_layer.save_memory(code)
        activation_decoder = torch.relu(activation_decoder)
        reconstructed = self.decoder_output_layer.save_memory(activation_decoder)
        reconstructed = torch.relu(reconstructed)
        
        #self.memory[0].append(features.cpu().numpy())
        #self.memory[1].append(activation.cpu().numpy())
        #self.memory[2].append(code.cpu().numpy())
        #self.memory[3].append(activation_decoder.cpu().numpy())
        #self.memory[4].append(reconstructed.cpu().numpy())
        
    '''
    grabs the weights from the layers to be saved out to a file
    '''
    def get_weights(self):
        return [self.encoder_hidden_layer.weight.cpu().detach().numpy(),
                self.encoder_output_layer.weight.cpu().detach().numpy(),
                self.decoder_hidden_layer.weight.cpu().detach().numpy(),
                self.decoder_output_layer.weight.cpu().detach().numpy()]
    
    '''
    uses the the memory in order to create a list of bins for the spiking
    network to use for discretization purposes
    '''
    def create_bins(self, bits):
        self.encoder_hidden_layer.create_bins(bits)
        self.encoder_output_layer.create_bins(bits)
        self.decoder_hidden_layer.create_bins(bits)
        self.decoder_output_layer.create_bins(bits)
    
 
# autoencoder spiking network
class AE_spikes(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
           of pretrained_model ~ use to recreate the weights exactly
    creates four layers, two for the encoder and two for the decoder, also 
    creates a totals array to hold the voltage of the neurons for each layer
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.totals = []
        self.layers = []
        
        # make the layers
        self.encoder_hidden_layer = SLinear(in_features=kwargs["input_shape"], 
                                            out_features=hidden_neurons)
        
        self.encoder_output_layer = SLinear(in_features=hidden_neurons, 
                                            out_features=hidden_neurons)
        
        self.decoder_hidden_layer = SLinear(in_features=hidden_neurons, 
                                            out_features=hidden_neurons)
        
        self.decoder_output_layer = SLinear(in_features=hidden_neurons, 
                                            out_features=kwargs["input_shape"])  
        
        pretrained_model = kwargs["pretrained_model"]
        self.copy(pretrained_model)
        
        # now append all the layers to a list and make totals for them
        self.layers.append(self.encoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.encoder_output_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.decoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.decoder_output_layer)
        self.totals.append(torch.zeros((1,kwargs["input_shape"]), device=device))
    
        
        self.create_input_totals = torch.zeros((1,kwargs["input_shape"]), device=device)
        
        self.totals = np.array(self.totals)
        
        self.set_up_weights()
        

    def copy(self, in_model):
        """ Convert might be a better name.
        """
        self.encoder_hidden_layer.copy(in_model.encoder_hidden_layer)
        self.encoder_output_layer.copy(in_model.encoder_output_layer)
        self.decoder_hidden_layer.copy(in_model.decoder_hidden_layer)
        self.decoder_output_layer.copy(in_model.decoder_output_layer)

        
    def set_up_weights(self):
        for layer in self.layers:
            # calculate how much voltage each spike causes in the neuron
            spike_height = layer.in_bins[1]-layer.in_bins[0]
            # multiply that into the weights
            layer.state_dict()['weight'] *= spike_height
            layer.state_dict()['bias'] *= spike_height
            
            # now do the same except for the output spike height
            spike_height = layer.out_bins[1]-layer.out_bins[0]
            # divide that into the weights
            layer.state_dict()['weight'] /= spike_height
            layer.state_dict()['bias'] /= spike_height
        

    '''
    uses a list of input spikes to calculate the output values of a layer
    the number in self.bits is the latency, max times the neuron will loop
    '''
    def process_layer(self, layer_num, input_spikes):
        # get the totals for this layer
        totals = self.totals[layer_num]

        # create the voltages in the cell from spikes
        total = self.layers[layer_num].state_dict()['weight'].clone().detach()*(input_spikes)
        totals += torch.sum(total,axis=1)
        
        # convert the voltage in the cell to spikes for output
        output = totals.clone().detach()
        output[output > 1] = 1
        output[output < 1] = 0
        
        # if neuron has spiked, subtract that from cell voltage
        totals[output == 1] -= 1
        
        
        return output 
    
    def process_first_layer(self, activations):
        # get the totals for this layer
        totals = self.create_input_totals
        
        # add the input volatges to the totals
        totals += activations

        # convert the voltage in the cell to spikes
        output = totals.clone().detach()
        output[output > 1] = 1
        output[output < 1] = 0

        # if the neuron spikes, subtract 1 from totals
        totals[output > 0] -= 1
        
        return output
    

    '''
    resets the totals to zero so that a new image can be processed
    '''
    def reset_totals(self):
        for idx in range(len(self.totals)):
            self.totals[idx] = self.layers[idx].state_dict()['bias'].clone().detach()

    '''
    a method updated from the super class, runs input through all the layers
    '''
    def forward(self, features):
        reconstructions = []
        # loop through every image in the batch
        for feature in features:
            # reset the totals to zero
            self.reset_totals()
            # divide input into 15
            # TODO: Try to get rid of this call.
            input_activation = self.encoder_hidden_layer.discretize(feature)/16.0

            output_spikes = torch.zeros(in_shape).to(device)
            
            # loop through every layer until we've finished processing the spikes
            for i in range(16):
                input_spikes = self.process_first_layer(input_activation)
                spike_layer1 = self.process_layer(0,input_spikes)
                spike_layer2 = self.process_layer(1,spike_layer1)
                spike_layer3 = self.process_layer(2,spike_layer2)
                output_spikes += self.process_layer(3,spike_layer3)

            # convert the output spikes to voltages
            reconstructed = self.decoder_output_layer.reconstruct(output_spikes)
            # add the reconstructed image to the list
            reconstructions.append(reconstructed)
            
        return reconstructions
    
    def forward_learn(self, features, layer):
        reconstructions = []

        # loop through every image in the batch
        for feature in features:
            # reset the totals to zero
            self.reset_totals()
            # divide input into 16
            input_activation = self.encoder_hidden_layer.discretize(feature)/16.0

            output_spikes = torch.zeros(in_shape).to(device)
            
            # create tensors that hold the spikes for learning later
            spike_frequencies = []
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(in_shape).to(device))
            # loop through every layer until we've finished processing the spikes
            for i in range(16):
                input_spikes = self.process_first_layer(input_activation)
                spike_layer1 = self.process_layer(0,input_spikes)
                spike_frequencies[0] += spike_layer1
                spike_layer2 = self.process_layer(1,spike_layer1)
                spike_frequencies[1] += spike_layer2
                spike_layer3 = self.process_layer(2,spike_layer2)
                spike_frequencies[2] += spike_layer3
                output_spikes += self.process_layer(3,spike_layer3)
            
            spike_frequencies[3] += output_spikes

            # change the spikes to frequencies
            for spike_frequency in spike_frequencies:
                spike_frequency /= 16.0
            
            # get the outputs from the original network for our truth
            og_layers = []
            og_layers.append(torch.relu(model.encoder_hidden_layer(feature)))
            og_layers.append(torch.relu(model.encoder_output_layer(og_layers[0])))
            og_layers.append(torch.relu(model.decoder_hidden_layer(og_layers[1])))
            og_layers.append(torch.relu(model.decoder_output_layer(og_layers[2])))

            # change to frequencies by using max bin
            og_layers[0] /= self.layers[0].out_bins[-1]
            og_layers[1] /= self.layers[1].out_bins[-1]
            og_layers[2] /= self.layers[2].out_bins[-1]
            og_layers[3] /= self.layers[3].out_bins[-1]
            
            # now we have to use the spike frequencies and og layer outputs to learn            
            if layer == 0:
                self.encoder_hidden_layer.weight += ((input_activation * 16) *\
                    torch.reshape((og_layers[0]-spike_frequencies[0]),[128,1])) * lr
                    
                self.encoder_hidden_layer.weight += (torch.ones(input_activation.shape).to(device) *\
                    torch.reshape((og_layers[0]-spike_frequencies[0]),[128,1])) * lr
                    
            else:
                self.layers[layer].weight += ((spike_frequencies[layer-1] * 16) *\
                    torch.reshape((og_layers[layer]-spike_frequencies[layer]),\
                                  [len(og_layers[layer]),1])) * lr
                    
                self.layers[layer].weight += (torch.ones(spike_frequencies[layer-1].shape).to(device) *\
                    torch.reshape((og_layers[layer]-spike_frequencies[layer]),\
                                  [len(og_layers[layer]),1])) * lr
            
            # convert the output spikes to voltages
            reconstructed = self.layers[3].reconstruct(output_spikes)
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
                                               batch_size=batch_size, 
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # mean-squared error loss
    criterion = nn.MSELoss()
    
    # train our autoencoder for our specified number of epochs.
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, in_shape] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, in_shape).to(device)
            
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
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))


def save_weights(model):
    np.save('weights_autoencoder',model.get_weights())
    return np.array(model.get_weights())

def load_weights():
    return np.load('weights_autoencoder.npy',allow_pickle=True)

def reconstruct_test_images(model, test_loader):
    test_examples = None

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, in_shape)
            reconstruction = model(test_examples.to(device))
            break
    
    # reconstruct some test images using trained autoencoder
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        

def compute_mMSE(model, test_loader):
    test_examples = None

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, in_shape)
            reconstruction = model(test_examples.to(device))
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


def train_spikingnet(model, train_loader, layer):
    test_examples = None

    # create the reconstructions using the model
    with torch.no_grad():
        for batch_features in train_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, in_shape)
            model.forward_learn(test_examples.to(device),layer)
            break
    

    
# save out the weights for the first layer, they look vaguely like parts of
# the handwritten numbers
def save_weight_images(weights):
    for idx in len(hidden_neurons):
        np.save('neuron_{}'.format(idx+1),weights[0][idx].reshape(28, 28))


# get the bins from the model using the test dataset
def compute_bins(model, test_loader, bits):
    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, in_shape)
            model.save_memory(test_examples.to(device))
            break
        
    model.create_bins(bits)
    
        

train_loader, test_loader = load_mnist()

# check if we've already trained this up
model = AE(input_shape=784).to(device)
try:
    with open('AEmodel.pth', 'rb') as config_dictionary_file:    
        model.load_state_dict(torch.load('AEmodel.pth',
                                         map_location=lambda storage, loc: storage))

except:
    # create a model from `AE` autoencoder class and load it to gpu    
    # train it
    train_autoencoder(model, train_loader)

    # save the model
    torch.save(model.state_dict(), 'AEmodel.pth')


# compute the mse
mMSE_AE = compute_mMSE(model, test_loader)


# for now a bit depth of 16
bits = 16
# get some bins for discretization purposes
compute_bins(model, test_loader, bits)

# create a spiking model
spiking_model = AE_spikes(input_shape=in_shape,
                          pretrained_model=model).to(device)


# compute the mse of the spiking model
mMSE_AE_spikes = compute_mMSE(spiking_model, test_loader)

reconstruct_test_images(model, test_loader)
reconstruct_test_images(spiking_model, test_loader)



print("Mean Squared Error of OG Model:")
print(mMSE_AE)

print("Mean Squared Error of Spiking Model:")
print(mMSE_AE_spikes)





for i in range(5):
    print("Epoch %d"%(i+1))
    for layer in range(4):
        print("Layer %d"%(layer+1))
        train_spikingnet(spiking_model, train_loader, layer)
        mMSE_AE_spikes_train = compute_mMSE(spiking_model, test_loader)

        print(mMSE_AE_spikes_train)
    




reconstruct_test_images(spiking_model, test_loader)



"""
mMSEs_spikes = []
bits_list = []

for bit_depth in range(2,64):
    bits = bit_depth
    bins = compute_bins(model, test_loader, bits)

    # create a spiking model
    spiking_model = AE_spikes(bits=bits,bins=bins,input_shape=in_shape,
                              pretrained_model=model).to(device)
    
    
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
