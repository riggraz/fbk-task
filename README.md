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

Results obtained by running the pipeline with seed=42 are the following:

```
--> SVM metrics
Accuracy = 0.97
Precision = 0.95
Recall = 0.98
F1 = 0.97
ROC AUC score = 0.98

--> Neural network metrics
Accuracy = 0.92
Precision = 0.90
Recall = 0.94
F1 = 0.92
ROC AUC score = 0.97

Performing McNemar test...
Contingency table
[[  59    9]
 [ 102 1830]]
pvalue      4.2608701445754604e-21
statistic   9.0
```

Since both models achieve high metrics, it can be said for certain that they were both able to learn and generalize quite well from the data. Obviously, it depends on the specific task which metrics are more important than other and whether the obtained results are acceptable or not.

Both models show a higher recall than precision, meaning that they both lean towards predicting "true" and having some false positives.

SVM has achieved overall better metrics than neural network, but the ROC AUC score is pretty similar between the two, meaning that they should have nearly the same ability to distinguish between the two classes. Choosing a different threshold for the neural network prediction may improve its metrics.

To sum up, SVM has achieved better results than neural network in every considered metric, and also the McNemar test has found a statistically significant difference in the number of errors of the two classifiers. It seems safe to say that, at least on the particular train-test split chosen and with the default 0.5 threshold, the SVM model should be preferred over the neural network one in terms of metrics.