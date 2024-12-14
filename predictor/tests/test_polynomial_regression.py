import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import logging

# Предполагаем, что ваши классы находятся в модуле your_module
from predictor.models import PolynomialRegressionModel, BaseMLModel  # Замените your_module на имя вашего модуля


# Фикстура для создания данных
@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    return df


class TestPolynomialRegressionModel:
    @pytest.fixture(autouse=True)
    def setup(self, sample_data):
        self.model = PolynomialRegressionModel(degree=2)
        self.model.load_data(sample_data, target_column='target')
        self.model.preprocess_data()
        self.model.split_data(test_size=0.2)

    def test_initialization(self):
        model = PolynomialRegressionModel(degree=3)
        assert model.degree == 3
        assert isinstance(model.model.named_steps['poly_features'], PolynomialFeatures)
        assert isinstance(model.model.named_steps['linear_regression'], LinearRegression)

    def test_load_data(self, sample_data):
        assert self.model.data_registry['raw_data'].equals(sample_data)
        assert self.model.target_column == 'target'

    def test_preprocess_data(self):
        assert self.model.data_registry['processed_data'] is not None

    def test_split_data(self):
        X_train, y_train = self.model.data_registry['train_data']
        X_test, y_test = self.model.data_registry['test_data']
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_fit(self):
        self.model.fit()
        assert self.model.model.named_steps['linear_regression'].coef_ is not None
        assert self.model.model.named_steps['linear_regression'].intercept_ is not None

    def test_predict(self):
        self.model.fit()
        predictions = self.model.predict()
        assert len(predictions) == 20
        assert 'mse' in self.model.metrics['test_scores']
        assert 'r2' in self.model.metrics['test_scores']

    def test_cross_validation(self):
        cv_scores = self.model.cross_validation(self.model.model, cv=5)
        assert len(cv_scores) == 5


if __name__ == "__main__":
    pytest.main()
