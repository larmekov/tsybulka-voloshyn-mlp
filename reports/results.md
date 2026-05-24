# Results

## Dataset

We used the Palmer Penguins dataset.

Missing values were removed using `dropna()`.  
Categorical features were converted into numerical features using one-hot encoding with `pandas.get_dummies()`.  
Numerical features were standardized using `StandardScaler`.

The dataset was split into training and test data with:

- test size: 0.2
- random state: 42

## Implemented Components

The following components were implemented from scratch using NumPy:

- multilayer perceptron
- forward pass
- backpropagation
- stochastic gradient descent parameter update
- mean squared error loss
- softmax cross-entropy loss
- evaluation metrics

The MLP uses two hidden layers with ReLU activation functions.

## Classification Task

The classification task was to predict the penguin species.

Target variable:

- `species`

Model architecture:

- input layer: 10 features
- hidden layer 1: 32 neurons, ReLU
- hidden layer 2: 16 neurons, ReLU
- output layer: 3 neurons

Training settings:

- loss function: cross-entropy
- optimizer: SGD
- epochs: 300
- learning rate: 0.05
- batch size: 32

Result:

| Model | Accuracy |
| NumPy MLP | 0.9851 |
| PyTorch MLP | 1.0000 |

## Regression Task

The regression task was to predict the body mass of a penguin in grams.

Target variable:

- `body_mass_g`

Model architecture:

- input layer: 10 features
- hidden layer 1: 32 neurons, ReLU
- hidden layer 2: 16 neurons, ReLU
- output layer: 1 neuron

Training settings:

- loss function: mean squared error
- optimizer: SGD
- epochs: 500
- learning rate: 0.01
- batch size: 32

The target variable was standardized during training and transformed back to grams for evaluation.

Result:

| Model | MAE | RMSE |
| NumPy MLP | 187.76 | 258.17 |
| PyTorch MLP | 190.06 | 251.47 |

The NumPy implementation and the PyTorch implementation achieved similar regression performance.

## Gradient Check

We compared gradients from backpropagation with numerical differentiation.

The numerical differentiation method used the forward difference formula:

```text
(f(x + h) - f(x)) / h
```
