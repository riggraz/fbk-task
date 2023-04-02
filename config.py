TEST_SET_SPLIT_SIZE = 0.2
SEED = 42

SVM_PARAM_GRID = {
  'C': [0.1, 1, 10, 100, 1000],
  'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
}

NN_PARAM_GRID = {
  'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
  'module__n_neurons': [20, 40, 60, 80, 100], # number of hidden layer neurons
}