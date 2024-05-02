import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import os

from fcn import FCNN
from dataset import Dataset as dtst


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


# Class that will containg the GN as well as methods for training, testing and predicting
class Formal(object):
    def __init__(self, batch_size=1, gpu=0, smooth=0.05, validation_split=0.25, datadir='../data/'):

        # Is a GPU available?
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        # Factor to be used for smoothing the loss with an exponential window
        self.smooth = smooth

        # If the nvidia_smi package is installed, then report some additional information
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(
                self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))

        self.batch_size = batch_size
        self.kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}

        self.datadir = datadir
        self.validation_split = validation_split

        # Instantiate the dataset
        self.dataset = dtst(self.datadir)

        # Randomly shuffle a vector with the indices to separate between training/validation datasets
        idx = np.arange(self.dataset.n_training)
        np.random.seed(666)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        print('-' * 50)
        print('Training samples: {0}'.format(len(self.train_index)))
        print('Validation samples: {0}'.format(len(self.validation_index)))
        print('-' * 50)

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        # Define the data loaders
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler, **self.kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.validation_sampler, **self.kwargs)

        # Instantiate the model with the hyperparameters
        self.model = FCNN(self.dataset.input_channels, self.dataset.n_classes).to(self.device)

        # Print the number of trainable parameters
        print('-' * 50)
        print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print('-'*50)


    def optimize(self, savedir, epochs, lr=1e-3):

        self.lr = lr
        self.n_epochs = epochs

        # Define the name of the model
        filename = time.strftime("%Y%m%d-%H%M%S")
        self.filename = filename
        self.savedir = savedir
        print('-' * 50)
        print('Model: {0}'.format(savedir + filename))
        print('-'*50)

        # if the savedir does not exist, create it
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 1, 0.01, self.n_epochs)

        # retrieve the class weights
        self.loss_weights = self.dataset.weights.to(self.device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=self.loss_weights, label_smoothing=self.smooth)

        # Now start the training
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):

            filename = time.strftime("%Y%m%d-%H%M%S")
            # Compute training and validation steps
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # If the validation loss improves, save the model as best
            if (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }

                print("Saving best model...")
                torch.save(checkpoint, savedir + filename + '_best.pth')
                self.best_filename = filename

            # Update the learning rate
            self.scheduler.step()

    def train(self, epoch):

        # Put the model in training mode
        self.model.train()
        print("\nEpoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0

        for batch_idx, data in enumerate(t):
            
            # Get the data
            x = data[0]
            y = data[1]

            # Move the data to the device
            x, y = x.to(self.device), y.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Evaluate Graphnet
            out = self.model(x)

            # Compute loss
            loss = self.loss_fn(out.squeeze(), y.squeeze())

            # Compute backpropagation
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            # Compute smoothed loss
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update information for this batch
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu,
                              memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        # Do a validation of the model and return the loss

        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):

                # Get the data
                x = data[0]
                y = data[1]

                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)

                loss = self.loss_fn(out.squeeze(), y.squeeze())

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                t.set_postfix(loss=loss_avg)

        return loss_avg
    
    def test(self):
        # Do a test of the model
        # collect the outputs and targets and return them

        # first restore the best model
        checkpoint = torch.load(self.savedir + self.best_filename + '_best.pth')
        print("Restoring best model: {0}".format(self.savedir + self.best_filename + '_best.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.eval()
        self.pred_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
        self.pred_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=self.pred_sampler, **self.kwargs)
        tq = tqdm(self.pred_loader)

        outputs = []
        targets = []
        with torch.no_grad():
            for data in tq:

                # Get the data
                x = data[0]
                y = data[1]

                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)

                # uncompress the tensors and pass it to cpu
                outputs.append(out.squeeze().cpu().numpy())
                targets.append(y.squeeze().cpu().numpy())

        return outputs, targets


if __name__ == '__main__':
    # Instantiate the Formal class
    formal = Formal(batch_size=4, gpu=0, smooth=0.05, validation_split=0.25, datadir='../data/')

    # Optimize the model
    formal.optimize(savedir='../models/', epochs=500, lr=1e-2)

    outputs, targets = formal.test()

    for i in range(len(outputs)):
        print(f'Output: {outputs[i]}, Target: {targets[i]}')
