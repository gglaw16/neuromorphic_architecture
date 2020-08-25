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
import pickle


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

lr = 1e-4


# use gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# autoencoder class with fully connected layers for both encoder and decoder
class AE(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
    creates four layers, two for the encoder and two for the decoder
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                              out_features=hidden_neurons)
        
        self.encoder_output_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_hidden_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_output_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=kwargs["input_shape"])
        
        self.memory = [[],[],[],[],[]]
        

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
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation_decoder = self.decoder_hidden_layer(code)
        activation_decoder = torch.relu(activation_decoder)
        reconstructed = self.decoder_output_layer(activation_decoder)
        reconstructed = torch.relu(reconstructed)
        
        self.memory[0].append(features.cpu().numpy())
        self.memory[1].append(activation.cpu().numpy())
        self.memory[2].append(code.cpu().numpy())
        self.memory[3].append(activation_decoder.cpu().numpy())
        self.memory[4].append(reconstructed.cpu().numpy())
        
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
        bins = []
        for layer_mem in self.memory:
            bins.append(np.histogram_bin_edges(layer_mem, bins=bits))
            
        self.bins = bins
        return bins
    
    
# autoencoder spiking network
class AE_spikes_latency(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
           of bits ~ use to tell how many bins there will be for discretization
           of bins ~ use to tell the values for the bins
           of pretrained_model ~ use to recreate the weights exactly
    creates four layers, two for the encoder and two for the decoder, also 
    creates a totals array to hold the voltage of the neurons for each layer
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.totals = []
        self.layers = []
        self.bits = kwargs["bits"]
        self.bins = kwargs["bins"]
        self.AEmodel = kwargs["pretrained_model"]
        
        self.encoder_hidden_layer = self.AEmodel.encoder_hidden_layer
        
        self.layers.append(self.encoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.encoder_output_layer = self.AEmodel.encoder_output_layer
        
        self.layers.append(self.encoder_output_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.decoder_hidden_layer = self.AEmodel.decoder_hidden_layer
        
        self.layers.append(self.decoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.decoder_output_layer = self.AEmodel.decoder_output_layer
        
        self.layers.append(self.decoder_output_layer)
        self.totals.append(torch.zeros((1,kwargs["input_shape"]), device=device))
        
        self.totals = np.array(self.totals)
        

    '''
    uses a list of input spikes to calculate the output values of a layer
    the number in self.bits is the latency, max times the neuron will loop
    '''
    def process_layer(self, layer_num, input_spikes):
        # get the totals for this layer
        totals = self.totals[layer_num]
        # calculate how much voltage each spike causes in the neuron
        spike_height = self.bins[layer_num][1]-self.bins[layer_num][0]
        # create a zero array to add up these voltages
        input_values = torch.zeros(input_spikes.shape)
        
        # while there are still spikes left over
        while len(input_spikes[input_spikes > 0]) > 0:
            # convert the first of the spikes left to voltages
            x = np.clip(input_spikes.cpu(),0,1).float()*spike_height
            # add those voltages to the total
            input_values += x
            # take those spikes off the list
            input_spikes[input_spikes > 0] -= 1
            
        # convert the voltages in the cell to output voltages
        totals = self.layers[layer_num](input_values.to(device))
    
           
        return np.clip(totals.cpu(),0,1000) # the clip is for relu
    
    
    '''
    uses the bins input in the init to change input values to a discrete number
    of spikes, uses idx to figure out which list of bins to use
    '''
    def discretize(self, idx, values):
        spikes = np.digitize(values.cpu(), self.bins[idx])-1
        return torch.from_numpy(spikes).to(device)

    '''
    resets the totals to zero so that a new image can be processed
    '''
    def reset_totals(self):
        for total in self.totals:
            total = np.clip(total.cpu(),0,0).to(device)

    '''
    a method updated from the super class, runs input through all the layers
    '''
    def forward(self, features):
        reconstructions = []
        # loop through every image in the batch
        for feature in features:
            # reset the totals to zero
            self.reset_totals()
            # run through every layer
            activation = self.discretize(0,feature)
            activation = self.process_layer(0,activation)
            code = self.discretize(1,activation)
            code = self.process_layer(1,code)
            activation_decoder = self.discretize(2,code)
            activation_decoder = self.process_layer(2,activation_decoder)
            reconstructed = self.discretize(3,activation_decoder)
            reconstructed = self.process_layer(3,reconstructed)
            # add the reconstructed image to the list
            reconstructions.append(reconstructed)
            
        return reconstructions
   
# autoencoder spiking network
class AE_spikes(nn.Module):
    '''
    kwargs of input_shape ~ use to tell the shape of the mnist image
           of bits ~ use to tell how many bins there will be for discretization
           of bins ~ use to tell the values for the bins
           of pretrained_model ~ use to recreate the weights exactly
    creates four layers, two for the encoder and two for the decoder, also 
    creates a totals array to hold the voltage of the neurons for each layer
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.totals = []
        self.layers = []
        self.bits = kwargs["bits"]
        self.bins = kwargs["bins"]
        
        # make the layers
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                              out_features=hidden_neurons)
        
        self.encoder_output_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_hidden_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=hidden_neurons)
        
        self.decoder_output_layer = nn.Linear(in_features=hidden_neurons, 
                                              out_features=kwargs["input_shape"])  
        
        self.load_state_dict(kwargs["pretrained_model"].state_dict())
        
        # now append all the layers to a list and make totals for them
        self.layers.append(self.encoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.encoder_output_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.decoder_hidden_layer)
        self.totals.append(torch.zeros((1,hidden_neurons), device=device))
        
        self.layers.append(self.decoder_output_layer)
        self.totals.append(torch.zeros((1,kwargs["input_shape"]), device=device))
        
        self.cont_to_disc_layer = nn.Linear(in_features=kwargs["input_shape"], 
                                              out_features=kwargs["input_shape"])
        
        self.ctdl_totals = torch.zeros((1,kwargs["input_shape"]), device=device)
        
        self.totals = np.array(self.totals)
        
        self.set_up_weights()
        
        
    def set_up_weights(self):
        for layer_num in range(len(self.layers)):
            # calculate how much voltage each spike causes in the neuron
            spike_height = self.bins[layer_num][1]-self.bins[layer_num][0]
            # multiply that into the weights
            self.layers[layer_num].state_dict()['weight'] *= spike_height
            self.layers[layer_num].state_dict()['bias'] *= spike_height
            
            # now do the same except for the output spike height
            spike_height = self.bins[layer_num+1][1]-self.bins[layer_num+1][0]
            # divide that into the weights
            self.layers[layer_num].state_dict()['weight'] /= spike_height
            self.layers[layer_num].state_dict()['bias'] /= spike_height
        

    '''
    uses a list of input spikes to calculate the output values of a layer
    the number in self.bits is the latency, max times the neuron will loop
    '''
    def process_layer(self, layer_num, input_spikes):
        # get the totals for this layer
        totals = self.totals[layer_num]

        # convert the first of the spikes left to voltages
        x = np.clip(input_spikes.cpu(),0,1).float()
        # take those spikes off the list
        input_spikes[input_spikes > 0] -= 1
            
        # convert the voltages in the cell to output voltages
        total = self.layers[layer_num].state_dict()['weight'].clone().detach()*(x.to(device))
        totals += torch.sum(total,axis=1)
        # perform relu filter
        output = np.clip(totals.cpu(),0,1).to(device)
        # convert the voltage in the cell to spikes

        output[output < 1] = 0
        totals[output == 1] -= 1
        
        # make sure there are no spikes if all the input was zero
        # this means if the totals is equal to the bias then set spike to 0
        output[totals==self.layers[layer_num].bias] = 0
        
        # if the neuron spikes, set totals to zero
        #totals[output > 0] = self.layers[layer_num].bias[output > 0]
        totals[output > 0] = 0
        
        return output 
    
    def process_first_layer(self, activations):
        # get the totals for this layer
        totals = self.ctdl_totals

        totals += activations

        # convert the voltage in the cell to spikes
        output = np.clip(totals.cpu(),0,1)

        # if the neuron spikes, set totals to zero
        totals[output > 0] = 0
        
        return output.to(device)
    
    '''
    uses the bins input in the init to change input values to a discrete number
    of spikes, uses idx to figure out which list of bins to use
    '''
    def discretize(self, idx, values):
        spikes = np.digitize(values.cpu(), self.bins[idx])-1
        return torch.from_numpy(spikes).to(device)

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
            input_activation = self.discretize(0,feature)/16.0
            spike_layers = []
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(784).to(device))
            
            # loop through every layer until we've finished processing the spikes
            for i in range(16):
                input_spikes = self.process_first_layer(input_activation)
                spike_layers[0] += self.process_layer(0,input_spikes)
                spike_layers[1] += self.process_layer(1,spike_layers[0])
                spike_layers[2] += self.process_layer(2,spike_layers[1])
                spike_layers[3] += self.process_layer(3,spike_layers[2])

            # convert the output spikes to voltages
            reconstructed = spike_layers[3]*(self.bins[4][1]-self.bins[4][0])
            # add the reconstructed image to the list
            reconstructions.append(reconstructed)
            
        return reconstructions
    
    def forward_learn(self, features, layer):
        reconstructions = []

        # loop through every image in the batch
        for feature in features:
            # reset the totals to zero
            self.reset_totals()
            # divide input into 15
            input_activation = self.discretize(0,feature)/16.0
            spike_layers = []
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(hidden_neurons).to(device))
            spike_layers.append(torch.zeros(784).to(device))
            
            spike_frequencies = []
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(hidden_neurons).to(device))
            spike_frequencies.append(torch.zeros(784).to(device))
            # loop through every layer until we've finished processing the spikes
            for i in range(16):
                input_spikes = self.process_first_layer(input_activation)
                output = self.process_layer(0,input_spikes)
                spike_layers[0] += output
                spike_frequencies[0] += output
                output = self.process_layer(1,spike_layers[0])
                spike_layers[1] += output
                spike_frequencies[1] += output
                output = self.process_layer(2,spike_layers[1])
                spike_layers[2] += output
                spike_frequencies[2] += output
                output = self.process_layer(3,spike_layers[2])
                spike_layers[3] += output
                spike_frequencies[3] += output

            for spike_frequency in spike_frequencies:
                spike_frequency /= 16.0
            
            og_layers = []
            og_layers.append(torch.relu(model.encoder_hidden_layer(feature)))
            og_layers.append(torch.relu(model.encoder_output_layer(og_layers[0])))
            og_layers.append(torch.relu(model.decoder_hidden_layer(og_layers[1])))
            og_layers.append(torch.relu(model.decoder_output_layer(og_layers[2])))
            
            og_layers[0] *= 16.0/self.bins[1][-1]
            og_layers[1] *= 16.0/self.bins[2][-1]
            og_layers[2] *= 16.0/self.bins[3][-1]
            og_layers[3] *= 16.0/self.bins[4][-1]
            
            for oglayer in og_layers:
                oglayer = torch.round(oglayer)
            
            og_layers[0] /= 16.0
            og_layers[1] /= 16.0
            og_layers[2] /= 16.0
            og_layers[3] /= 16.0
            
            # now we have to use the spike frequencies and og layer outputs to learn
            
            error = torch.reshape((og_layers[0]-spike_frequencies[0]),[128,1])
            #error = error*error
            print(error.mean())
            
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
            reconstructed = spike_layers[3]*(self.bins[4][1]-self.bins[4][0])
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
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)
            
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
            test_examples = batch_features.view(-1, 784)
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
            test_examples = batch_features.view(-1, 784)
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
            test_examples = batch_features.view(-1, 784)
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
            test_examples = batch_features.view(-1, 784)
            model.save_memory(test_examples.to(device))
            break
        
    return model.create_bins(bits)
    
# saves out any object to a file
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        

train_loader, test_loader = load_mnist()

# check if we've already trained this up
try:
    with open('AEmodel.pkl', 'rb') as config_dictionary_file:
    
        model = pickle.load(config_dictionary_file)

except:
    # create a model from `AE` autoencoder class and load it to gpu
    model = AE(input_shape=784).to(device)
    
    # train it
    train_autoencoder(model, train_loader)
    
    # save the model
    save_object(model, 'AEmodel.pkl')


# compute the mse
mMSE_AE = compute_mMSE(model, test_loader)


# for now a bit depth of 16
bits = 16
# get some bins for discretization purposes
bins = compute_bins(model, test_loader, bits)

# create a spiking model
spiking_model = AE_spikes(bits=bits,bins=bins,input_shape=784,
                          pretrained_model=model).to(device)


# compute the mse of the spiking model
mMSE_AE_spikes = compute_mMSE(spiking_model, test_loader)

#reconstruct_test_images(model, test_loader)
#reconstruct_test_images(spiking_model, test_loader)



print("Mean Squared Error of OG Model:")
print(mMSE_AE)

print("Mean Squared Error of Spiking Model:")
print(mMSE_AE_spikes)




lastmse = 1

for i in range(1):
    for layer in range(1):
        print("Layer %d"%(layer+1))
        train_spikingnet(spiking_model, train_loader, layer)
        mMSE_AE_spikes_train = compute_mMSE(spiking_model, test_loader)

        print(mMSE_AE_spikes_train)




#reconstruct_test_images(spiking_model, test_loader)

"""
mMSEs_spikes = []
bits_list = []

for bit_depth in range(2,64):
    bits = bit_depth
    bins = compute_bins(model, test_loader, bits)

    # create a spiking model
    spiking_model = AE_spikes(bits=bits,bins=bins,input_shape=784,
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
