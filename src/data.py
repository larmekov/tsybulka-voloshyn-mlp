import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(task="classification", test_size=0.2, random_state=42):
    df = load_penguins()
    df = df.dropna()

    if task == "classification":
        y = df["species"].map({
            "Adelie": 0,
            "Chinstrap": 1,
            "Gentoo": 2,
        }).values

        X = df.drop(columns=["species"])

    elif task == "regression":
        y = df["body_mass_g"].values
        X = df.drop(columns=["body_mass_g"])

    else:
        raise ValueError("task must be 'classification' or 'regression'")

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test