import pytest
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from predictor.models import DecisionTreeModel, BaseMLModel  # Замените your_module на имя вашего модуля


# Фикстура для создания данных регрессии
@pytest.fixture(scope="class")
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    return df


# Фикстура для создания данных классификации
@pytest.fixture(scope="class")
def classification_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    return df


class TestDecisionTreeModel:
    @pytest.fixture(autouse=True)
    def setup(self, regression_data, classification_data):
        self.reg_model = DecisionTreeModel(task_type='regression')
        self.reg_model.load_data(regression_data, target_column='target')
        self.reg_model.preprocess_data()
        self.reg_model.split_data(test_size=0.2)

        self.clf_model = DecisionTreeModel(task_type='classification')
        self.clf_model.load_data(classification_data, target_column='target')
        self.clf_model.preprocess_data()
        self.clf_model.split_data(test_size=0.2)

    def test_initialization_regression(self):
        model = DecisionTreeModel(
            task_type='regression',
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2
        )
        assert model.task_type == 'regression'
        assert model.model_params['max_depth'] == 3
        assert model.model_params['min_samples_split'] == 5
        assert model.model_params['min_samples_leaf'] == 2
        assert model.model_params['criterion'] == 'squared_error'
        assert isinstance(model.model, DecisionTreeRegressor)

    def test_initialization_classification(self):
        model = DecisionTreeModel(
            task_type='classification',
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2
        )
        assert model.task_type == 'classification'
        assert model.model_params['max_depth'] == 3
        assert model.model_params['min_samples_split'] == 5
        assert model.model_params['min_samples_leaf'] == 2
        assert model.model_params['criterion'] == 'gini'
        assert isinstance(model.model, DecisionTreeClassifier)

    def test_load_data_regression(self, regression_data):
        assert self.reg_model.data_registry['raw_data'].equals(regression_data)
        assert self.reg_model.target_column == 'target'

    def test_load_data_classification(self, classification_data):
        assert self.clf_model.data_registry['raw_data'].equals(classification_data)
        assert self.clf_model.target_column == 'target'

    def test_preprocess_data_regression(self):
        assert self.reg_model.data_registry['processed_data'] is not None

    def test_preprocess_data_classification(self):
        assert self.clf_model.data_registry['processed_data'] is not None

    def test_split_data_regression(self):
        X_train, y_train = self.reg_model.data_registry['train_data']
        X_test, y_test = self.reg_model.data_registry['test_data']
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_split_data_classification(self):
        X_train, y_train = self.clf_model.data_registry['train_data']
        X_test, y_test = self.clf_model.data_registry['test_data']
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_fit_regression(self):
        self.reg_model.fit()
        assert self.reg_model.feature_importances is not None

    def test_fit_classification(self):
        self.clf_model.fit()
        assert self.clf_model.feature_importances is not None

    def test_predict_regression(self):
        self.reg_model.fit()
        predictions = self.reg_model.predict()
        assert len(predictions) == 20
        assert 'mse' in self.reg_model.metrics['test_scores']
        assert 'r2' in self.reg_model.metrics['test_scores']

    def test_predict_classification(self):
        self.clf_model.fit()
        predictions = self.clf_model.predict()
        assert len(predictions) == 20
        assert 'accuracy' in self.clf_model.metrics['test_scores']
        assert 'report' in self.clf_model.metrics['test_scores']

    def test_cross_validation_regression(self):
        cv_scores = self.reg_model.cross_validation(self.reg_model.model, cv=5)
        assert len(cv_scores) == 5

    def test_cross_validation_classification(self):
        cv_scores = self.clf_model.cross_validation(self.clf_model.model, cv=5)
        assert len(cv_scores) == 5


if __name__ == "__main__":
    pytest.main()
