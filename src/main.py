from train import train_classification, train_regression
from grad_check import compare_gradients
from pytorch_check import train_torch_classification, train_torch_regression


if __name__ == "__main__":
    print("NumPy MLP training")
    print("=================")

    print()
    print("Classification")
    train_classification()

    print()
    print("Regression")
    train_regression()

    print()
    print("Gradient check")
    print("==============")
    compare_gradients(1e-1)
    compare_gradients(1e-3)
    compare_gradients(1e-5)

    print()
    print("PyTorch comparison")
    print("==================")
    train_torch_classification()
    train_torch_regression()