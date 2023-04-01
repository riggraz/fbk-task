from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import np2tensor

# This class performs grid search, training and evaluation of given model
class ModelPipeline:
  def __init__(self, model, dataset, param_grid, seed=None, convert_data_to_tensor=False) -> None:
    self.model = model
    self.dataset = dataset
    self.param_grid = param_grid
    self.seed = seed
    self.convert_data_to_tensor = convert_data_to_tensor

  def run(self):
    X_train, X_test, y_train, y_test = self.dataset

    # Define cross validation strategy
    cv = KFold(n_splits=3, shuffle=True, random_state=self.seed)

    # Define grid search
    grid_search = GridSearchCV(
      self.model,
      param_grid=self.param_grid,
      cv=cv,
      refit=True,
      n_jobs=1,
      verbose=2,
    )

    # Perform grid search and retraining with best hyperparameters
    print(f'Performing grid search with param grid: {self.param_grid}')
    if self.convert_data_to_tensor:
      grid_search_result = grid_search.fit(np2tensor(X_train), np2tensor(y_train))
    else:
      grid_search_result = grid_search.fit(X_train, y_train)

    # Get best classifier
    print(f'Best hyperparameters: {grid_search_result.best_params_}')
    classifier = grid_search_result.best_estimator_

    # Evaluate classifier
    print(f'Evaluating classifier...')
    y_test_pred = classifier.predict(np2tensor(X_test) if self.convert_data_to_tensor else X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f'Accuracy = {accuracy:.2f}')
    print(f'Precision = {precision:.2f}')
    print(f'Recall = {recall:.2f}')
    print(f'F1 = {f1:.2f}')

    return accuracy, precision, recall, f1
