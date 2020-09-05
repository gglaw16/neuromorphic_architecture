import numpy as np
import torch
import torch.nn as nn

#import ipdb


# Notes:  I could execute a series of spikes in one forward call,
# but I want to support batches api like normal forward methods.
# Should we change output ot int or bool?
# Smaller memory might allow for batches and trains of spikes.
# No, we still need to store the potentials.

# Subclasses of torch.nn layers that have extra functionality for spiking.


# encoding (analog to spiking) layer
class Encoder(nn.Identity):
    def __init__(self, num_features):
        super().__init__(num_features)        

    def get_spike_frequencies(self):
        return self.spike_counts / self.frequency_duration

    def reset(self):
        """ Reset membrane potential and counts to start computing spikes and frequency a new.
        """
        # I am dealying the allocation of the integration tensors until execution.
        # We do not know the shape necessary for these tensors until we know the shape of the input / output.
        self.potentials = None
        self.spike_counts = None
        self.frequency_duration = 0

        
    def translate_weights(self):
        """ spike encoder layer has no weights to translate
        """
        pass

    
    def process(self, activations):
        """ Activations are continuous values, returns output spikes (binary values).
        """
        # Initialize the integration variables.
        if self.potentials is None:
            self.potentials = torch.zeros(activations.shape,
                                          device=activations.device)
        if self.spike_counts is None:
            self.spike_counts = torch.zeros(activations.shape,
                                            device=activations.device)

            self.frequency_duration = 0

        potentials = self.potentials
        # add the input currents to the membrane potentials (capacitence)
        potentials += activations
        # convert the voltage in the cell to spikes
        output = torch.zeros(potentials.shape, device=potentials.device)
        output[potentials > 1] = 1
        # if the neuron spikes, subtract 1 from potentials
        potentials[output > 0] -= 1

        # For learning.
        self.spike_counts += output
        self.frequency_duration += 1
        
        return output
    




# spiking linear layer
class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features)        
        # There is duplication of output/inputs bwtween layers, but
        # this allows mamangent of memeory and bins in layers.
        # An alternative is to create a spike encoding layer to manage input bins.
        self.in_memory = []
        self.in_bins = []
        self.out_memory = []        
        self.out_bins = []
        # I am dealying the allocation of the integration tensors until execution.
        # We do not know the shape necessary for these tensors until we know the shape of the input / output.
        self.potentials = None
        self.spike_counts = None
        self.frequency_duration = 0


    def translate_weights(self):
        # calculate how much voltage each spike causes in the neuron
        spike_height = self.in_bins[1]-self.in_bins[0]
        # multiply that into the weights
        self.state_dict()['weight'] *= spike_height
        self.state_dict()['bias'] *= spike_height
            
        # now do the same except for the output spike height
        spike_height = self.out_bins[1]-self.out_bins[0]
        # divide that into the weights
        self.state_dict()['weight'] /= spike_height
        self.state_dict()['bias'] /= spike_height
        self.state_dict()['bias'] /= 16.0
        
        
    # TODO: Overwrite superclass execute layer method
    def process(self, input_spikes):
        """ Uses a list of input spikes to calculate the output values of a layer.
        """

        # create the voltages in the cell from input spikes (currents)
        input_current = self(input_spikes)
        
        # get the potentials for this layer
        if self.potentials is None:
            self.potentials = torch.zeros(input_current.shape, device=input_current.device)

        # Integrate current witn membrane capacitance to get voltage.
        self.potentials += input_current
        
        # convert the voltage in the cell to spikes for output
        output = torch.zeros(input_current.shape, device=input_current.device)
        output[self.potentials > 1] = 1
        
        # if neuron has spiked, subtract that from cell voltage
        self.potentials[output == 1] -= 1
        
        # For learning.
        if self.spike_counts is None:
            self.spike_counts = torch.zeros(output.shape,
                                            device=output.device)
        self.spike_counts += output
        self.frequency_duration += 1

        return output 


    def get_spike_frequencies(self):
        return self.spike_counts / self.frequency_duration

    
    def reset(self):
        """ Reset membrane potential and counts to start computing spikes and frequency a new.
        """
        self.potentials = None
        self.spike_counts = None
        self.frequency_duration = 0

        
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

    def get_activation(self):
        """ Get a continuous activation that is equivalaent spiking rate.
        Uses bins to try and reverse translate firing rate.
        I do not think the logic is right.
        """
        return self.spike_counts*(self.out_bins[1]-self.out_bins[0])








# spiking linear layer
# I do not think bias is handled correctly since it is not used after every spike.
# Right now it is only used for initialization.
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1)        
        # There is duplication of output/inputs bwtween layers, but
        # this allows mamangent of memeory and bins in layers.
        # An alternative is to create a spike encoding layer to manage input bins.
        self.in_memory = []
        self.in_bins = []
        self.out_memory = []        
        self.out_bins = []
        # I am dealying the allocation of the integration tensors until execution.
        # We do not know the shape necessary for these tensors until we know the shape of the input / output.
        self.potentials = None
        self.spike_counts = None
        self.frequency_duration = 0


    def translate_weights(self):
        # calculate how much voltage each spike causes in the neuron
        spike_height = self.in_bins[1]-self.in_bins[0]
        # multiply that into the weights
        self.state_dict()['weight'] *= spike_height
        layer_bias = self.state_dict()['bias']
        self.spike_bias = layer_bias * spike_height
        layer_bias.fill_(0.0)
        
        # now do the same except for the output spike height
        spike_height = self.out_bins[1]-self.out_bins[0]
        # divide that into the weights
        self.state_dict()['weight'] /= spike_height
        self.spike_bias /= spike_height
        
        
    # TODO: Overwrite superclass execute layer method
    def process(self, input_spikes):
        """ Uses input spikes to calculate the output values of a layer.
        """

        # create the voltages in the cell from input spikes (currents)
        total = self(input_spikes)

        # Initialize if necessary
        if self.potentials is None:
            self.potentials = torch.zeros(total.shape,
                                          device=total.device)
            # Broadcast the biases to initializ the potential
            self.potentials = self.potentials + self.spike_bias 

        self.potentials += torch.sum(total,axis=1)
        
        # convert the voltage in the cell to spikes for output
        output = self.potentials.clone().detach()
        output[output > 1] = 1
        output[output < 1] = 0
        
        # if neuron has spiked, subtract that from cell voltage
        self.potentials[output == 1] -= 1
        
        # For learning.
        if self.spike_counts is None:
            self.spike_counts = torch.zeros(output.shape,
                                            device=output.device)
        self.spike_counts += output
        self.frequency_duration += 1

        return output 


    def get_spike_frequencies(self):
        return self.spike_counts / self.frequency_duration

    
    def reset(self):
        """ Reset membrane potential and counts to start computing spikes and frequency a new.
        """
        self.potentials = None
        self.spike_counts = None
        self.frequency_duration = 0

        
    def save_memory(self, x):
        """ This is an alternative execute function that executes in continuous mode 
        and also records input and output activation functions.
        This is required to chose a discritization for spking.
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


    # TODO: Try to get rid of this method.
    def discretize(self, values):
        """
        uses the bins input in the init to change input values to a discrete number
        of spikes, uses idx to figure out which list of bins to use
        """
        spikes = np.digitize(values.cpu(), self.in_bins)-1
        return torch.from_numpy(spikes).to(values.device)

    
    def reconstruct(self, spike_frequencies):
        spike_counts = spike_frequencies * self.frequency_duration
        return spike_counts*(self.out_bins[1]-self.out_bins[0])

        
        

    
        

