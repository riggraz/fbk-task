## Requirements

- Python (tested with 3.8.5)

## Installation

Run the following commands inside the project folder:

- Create a venv: `python -m venv .venv`
- Activate the venv: `source .venv/bin/activate` (Unix or macOS) or `.venv\Scripts\activate.bat` (Windows)
- Install required packages: `pip install -r requirements.txt`

## Run

- Run the project: `python main.py`

## Pipeline description

The dataset has been split into training and test parts, so results are bound to the specific train-test split performed. It could have been possible to perform an external cross validation in order to get more reliable results (i.e. not dependend on the specific train-test split), but at the cost of a lot more computational time.

The compared models are a SVM and a neural network.

The grid search algorithm has been used for hyperparameter tuning. In particular, a 3-fold cross validation has been performed on the training set for each hyperparameter combination. It has been decided to split between training and validation sets in order to not overfit hyperparameters to the test set. Then, the best hyperparameters have been used to train the models over the entire training set.

Finally, the obtained classifiers has been evaluated in terms of accuracy, precision, recall, F1 and AUC, and have been compared with the McNemar statistical test.

## Architectural choices for the code

The whole pipeline logic is contained in a single class (`Pipeline`). The grid search and training logic has been extracted into a separate class (`TrainingPipeline`) for reusability purposes across the two models.

Configurations for train-test split size, seed and hyperparameters grids can be found in the file `config.py`.

## Libraries

- PyTorch: used to build the neural network model
- Scikit Learn: used for SVM model, fitting models, grid search and metrics computation
- Skorch: used to wrap the PyTorch neural network in order to use it with Scikit Learn
- Statsmodels: used to compute McNemar statistic

## Results

