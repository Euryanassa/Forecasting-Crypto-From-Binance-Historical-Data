import numpy as np
import pandas as pd
import jinja2
from pycaret.regression import *
import datetime



class statistical_models():
    def __init__(self, model_id):
      self.model_id = model_id

    def model_creator(self, df, save_path = '', target = 'y'):
        try:
            print(f'[\033[96mLOG\033[0m]Enviroment Creation Begins')
            setup(df, target = 'y', silent = True, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Enviroment Has Created Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Enviroment Setup Error! All be demolished soon!')
        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Creation Begins')
            model = create_model(self.model_id, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Created Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Not Created Successfully!')
        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Tuning Begins')
            tuned_model = tune_model(model, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Tuning Process Has Ended Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Couldn\'t Tuned Successfully')

        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Tuned Version Saving...')
            now = datetime.datetime.now()
            save_model(tuned_model,
                      f'{save_path}{self.model_id}_model_{now.year}_{now.month}_{now.day}-{now.hour}:{now.minute}:{now.second}',
                      verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Saved Successfully!')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Couldn\'t Saved')

    def model_pool():
        models = {'lr':'Linear Regression',
                  'lasso': 'Lasso Regression',
                  'ridge': 'Ridge Regression',
                  'en': 'Elastic Net',
                  'lar': 'Least Angle Regression',
                  'llar': 'Lasso Least Angle Regression',
                  'omp': 'Orthogonal Matching Pursuit',
                  'br': 'Bayesian Ridge',
                  'ard': 'Automatic Relevance Determination',
                  'par': 'Passive Aggressive Regressor',
                  'ransac': 'Random Sample Consensus',
                  'tr': 'TheilSen Regressor',
                  'huber': 'Huber Regressor',
                  'kr': 'Kernel Ridge',
                  'svm': 'Support Vector Regression',
                  'knn': 'K Neighbours Regressor',
                  'dt': 'Decisiopn Tree Regressor',
                  'rf': 'Random Forest Regressor',
                  'et': 'Extra Trees Regressor',
                  'ada': 'AdaBoost Regressor',
                  'gbr': 'Gradient Boosting Regressor',
                  'mlp': 'MLP Regressor',
                  'xgboost': 'Extreme Gradient Boosting',
                  'lightgbm': 'Light Gradient Boosting Machine',
                  'catboost': 'CatBoost Regressor'}
        return models
