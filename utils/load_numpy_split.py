import numpy as np

def load_numpy_and_split(numpy_file, split_indice=132720):
    data = np.load(numpy_file)
    train_numpy, test_numpy = data[:split_indice], data[split_indice:]
    return train_numpy, test_numpy
