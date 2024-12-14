import pickle
import pandas as pd
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import numpy as np

class ModelPredictor:
    def __init__(self, model_path: str, preprocessing_func=None):
        """
        model_path: путь к заранее обученной модели (pickle-файл).
        preprocessing_func: функция для предобработки данных перед предсказанием.
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.preprocessing_func = preprocessing_func

    def _load_model(self):
        """Загрузка модели из pickle-файла"""
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model