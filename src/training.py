import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

import scaling

class Model:
    def __init__(self, model: BaseEstimator, df: pd.DataFrame, target: str):
        self.model = model
        self.df = df
        self.target = target
        self.data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = None
        self.scaled_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = None

    def split(self):
        X = self.df.drop(columns=[self.target])
        Y = self.df[self.target]

        self.data = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.scaled_data = self.data

    def scale(self, transformer: TransformerMixin):
        if self.data is None: raise ValueError("Data must be split first.")

        X_train, X_test, Y_train, Y_test = self.data
        X_train_norm, X_test_norm = scaling.normalize(X_train, X_test, transformer)

        self.scaled_data = (X_train_norm, X_test_norm, Y_train, Y_test)

    def train(self):
        if self.data is None: raise ValueError("Data must be split first.")

        X_train, _, Y_train, _ = self.scaled_data
        self.model.fit(X_train, Y_train)

    def score(self):
        if self.data is None: raise ValueError("Data must be split first.")

        _, X_test, _, Y_test = self.scaled_data
        return self.model.score(X_test, Y_test)

    def predict(self):
        if self.data is None: raise ValueError("Data must be split first.")

        _, X_test, _, _ = self.scaled_data
        return self.model.predict(X_test)

    def metrics(self) -> dict:
        if self.data is None: raise ValueError("Data must be split first.")

        _, _, _, Y_test = self.scaled_data
        Y_test_pred = self.predict()
        score = self.score()

        if isinstance(self.model, ClassifierMixin):
            precision = precision_score(Y_test, Y_test_pred, average="micro")
            recall = recall_score(Y_test, Y_test_pred, average="micro")
            return { "accuracy": score, "precision": precision, "recall": recall }
        else:
            mae = mean_absolute_error(Y_test_pred, Y_test)
            rmse = root_mean_squared_error(Y_test_pred, Y_test)
            return { "r2": score, "mae": mae, "rmse": rmse }
