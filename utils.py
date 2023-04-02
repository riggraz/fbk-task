import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the 'object' dataset and returns it
def load_object_dataset(dataset_name='object.csv'):
  dataset = np.genfromtxt(
    dataset_name,
    dtype=np.float64,
    delimiter=',',
    skip_header=1,
    converters={0: lambda label: 0.0 if label.decode() == 'object1' else 1.0},
  )

  return dataset

# Converts a numpy array to a torch tensor (float)
def np2tensor(np_array):
  return torch.tensor(np_array, dtype=torch.float32)

# Prints evaluation metrics
def print_metrics(y_true, y_pred) -> None:
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)

  print(f'Accuracy = {accuracy:.2f}')
  print(f'Precision = {precision:.2f}')
  print(f'Recall = {recall:.2f}')
  print(f'F1 = {f1:.2f}')