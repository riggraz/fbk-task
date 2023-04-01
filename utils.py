import numpy as np
import torch

def load_object_dataset(dataset_name='object.csv'):
  dataset = np.genfromtxt(
    dataset_name,
    dtype=np.float64,
    delimiter=',',
    skip_header=1,
    converters={0: lambda label: 0.0 if label.decode() == 'object1' else 1.0},
  )

  return dataset

def np2tensor(np_array):
  return torch.tensor(np_array, dtype=torch.float32)