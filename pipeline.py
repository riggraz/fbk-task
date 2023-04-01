import torch
import skorch
from sklearn import svm
from sklearn.model_selection import train_test_split

import config
from utils import load_object_dataset
from model_pipeline import ModelPipeline
from nn import NN

class Pipeline:
  def __init__(self) -> None:
    self.dataset = load_object_dataset()

  def run(self) -> None:
    # Split between features and labels
    X, y = self.dataset[:, 1:], self.dataset[:, 0]

    # Split in train and test set
    X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=config.TEST_SET_SPLIT_SIZE,
      random_state=config.SEED,
    )

    # Set up and run SVM pipeline
    print('--> SVM')

    svm_model = svm.SVC()

    svm_pipeline = ModelPipeline(
      svm_model,
      (X_train, X_test, y_train, y_test),
      config.SVM_PARAM_GRID,
      seed=config.SEED,
    )
    svm_pipeline.run()

    # Set up and run neural network pipeline
    print('\n\n--> Neural network')

    # Wrap PyTorch model in skorch to use it with sklearn
    nn_model = skorch.NeuralNetBinaryClassifier(
      NN,
      criterion=torch.nn.BCEWithLogitsLoss,
      batch_size=50,
      max_epochs=10,
      train_split=None, # do not use validation set (it's already done by GridSearchCV in the model pipeline)
      verbose=0,
    )

    nn_pipeline = ModelPipeline(
      nn_model,
      (X_train, X_test, y_train, y_test),
      config.NN_PARAM_GRID,
      seed=config.SEED,
      convert_data_to_tensor=True,
    )
    nn_pipeline.run()
