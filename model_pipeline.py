from sklearn.model_selection import KFold, GridSearchCV

from utils import np2tensor

# Performs grid search and returns best classifier
class ModelPipeline:
  def __init__(self, model, train_dataset, param_grid, seed=None, convert_data_to_tensor=False) -> None:
    self.model = model
    self.train_dataset = train_dataset
    self.param_grid = param_grid
    self.seed = seed
    self.convert_data_to_tensor = convert_data_to_tensor

  def run(self):
    X_train, y_train = self.train_dataset

    # Define cross validation strategy
    cv = KFold(n_splits=3, shuffle=True, random_state=self.seed)

    # Define grid search
    grid_search = GridSearchCV(
      self.model,
      param_grid=self.param_grid,
      cv=cv,
      refit=True, # retrain best model over entire training set
      n_jobs=1,
      verbose=3,
    )

    # Perform grid search and retraining with best hyperparameters
    print(f'Performing grid search with param grid: {self.param_grid}')
    if self.convert_data_to_tensor:
      grid_search_result = grid_search.fit(np2tensor(X_train), np2tensor(y_train))
    else:
      grid_search_result = grid_search.fit(X_train, y_train)

    print(f'Best hyperparameters: {grid_search_result.best_params_}')
    
    # Return best classifier
    return grid_search_result.best_estimator_
