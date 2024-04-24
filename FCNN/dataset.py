import torch
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir='../data/'):
        """
        Dataset for the depth stratification
        """
        super(Dataset, self).__init__()

        # Read the training database
        with open(datadir + 'dataset.pkl', 'rb') as filehandle:
            self.dataset_raw = pickle.load(filehandle)

        self.x = torch.tensor(self.dataset_raw['x'], dtype=torch.float32)
        self.y = torch.tensor(self.dataset_raw['y'], dtype=torch.long)

        # Number of training samples
        self.n_training = self.x.shape[0]

        # create a list with the classes in the dataset
        self.classes = list(set(self.y.tolist()))
        # Number of classes
        self.n_classes = len(self.classes)

        # compute how many times each class appears in the dataset
        self.class_count = [0] * self.n_classes
        for i in range(self.n_training):
            self.class_count[self.y[i]] += 1
        self.class_count = torch.tensor(self.class_count, dtype=torch.float32)
        
        # create the weigths as the inverse of the class count
        self.weights = 1.0 / self.class_count
        # and normalize the weights to 1
        self.weights = self.weights / torch.sum(self.weights)

        # Number of input channels
        self.input_channels = self.x[0].shape[0]

        print('-' * 50)
        print('  Dataset information')
        print(f'  Number of samples: {self.n_training}')
        print(f'  Number of classes: {self.n_classes}')
        print(f'  Number of input channels: {self.input_channels}')
        print(f'  Classes: {self.classes}')
        print(f'  Class count: {self.class_count}')
        print(f'  Weights: {self.weights}')
        print('-' * 50)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_training

    def __call__(self, index):
        return self.x[index], self.y[index]
