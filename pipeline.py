import torch
import skorch
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix
from statsmodels.stats.contingency_tables import mcnemar

import config
from utils import load_object_dataset, np2tensor, print_metrics
from training_pipeline import TrainingPipeline
from nn import NN

torch.manual_seed(config.SEED)

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

    # Set up and run SVM training pipeline
    print('--> SVM')

    svm_pipeline = TrainingPipeline(
      svm.SVC(),
      (X_train, y_train),
      config.SVM_PARAM_GRID,
      seed=config.SEED,
    )
    svm_classifier = svm_pipeline.run()

    # Set up and run neural network training pipeline
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

    nn_pipeline = TrainingPipeline(
      nn_model,
      (X_train, y_train),
      config.NN_PARAM_GRID,
      seed=config.SEED,
      convert_data_to_tensor=True,
    )
    nn_classifier = nn_pipeline.run()

    # Evaluate the classifiers
    svm_y_test_pred = svm_classifier.predict(X_test)
    print('--> SVM metrics')
    print_metrics(y_test, svm_y_test_pred)

    nn_y_test_pred = nn_classifier.predict(np2tensor(X_test))
    print('--> Neural network metrics')
    print_metrics(y_test, nn_y_test_pred)

    # Compare the two classifiers with McNemar's test
    print('Performing McNemar test...')
    svm_y_test_pred_correctness = svm_y_test_pred == y_test
    nn_y_test_pred_correctness = nn_y_test_pred == y_test
    contingency_table = contingency_matrix(svm_y_test_pred_correctness, nn_y_test_pred_correctness)
    print(f'Contingency table = {contingency_table}')

    mcnemar_result = mcnemar(contingency_table)
    print(mcnemar_result)

    alpha = 0.05
    if mcnemar_result.pvalue > alpha:
      print('Null hypothesis cannot be rejected. The two models have NO meaningfully different error rates on test set.')
    else:
      print('Null hypothesis can be rejected. The two models have meaningfully different error rates on the test set.')
    