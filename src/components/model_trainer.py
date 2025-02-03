import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGB Regressor': XGBRegressor(),
                'Catboost Regressor': CatBoostRegressor(verbose=False),
                'Adaboost Regressor': AdaBoostRegressor(),
                'Support Vector Regressor': SVR()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            if not model_report:
                raise CustomException('No models trained successfully')

            # Ambil model terbaik berdasarkan r2_test
            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_test'])
            best_model = models[best_model_name]
            best_r2_train = model_report[best_model_name]['r2_train']
            best_r2_test = model_report[best_model_name]['r2_test']

            if best_r2_test < 0.6:
                raise CustomException('No best model found')

            logging.info(f'Best model: {best_model_name} | R² Train: {best_r2_train:.4f} | R² Test: {best_r2_test:.4f}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_r2_train, best_r2_test

        except Exception as e:
            raise CustomException(e, sys)