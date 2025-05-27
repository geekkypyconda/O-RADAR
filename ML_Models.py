from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, log_loss,make_scorer
)

from sklearn.linear_model import(
    LogisticRegression
)

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier
from ORAN_Helper import Metric
import joblib as jlb
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

import h5py

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("\n\n\n<<<<<<<<<<<<<<----------------------------------->>>>>>>>>>>>>>>>>>>")
print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
print("<<<<<<<<<<<<<<----------------------------------->>>>>>>>>>>>>>>>>>>\n\n\n")



class LR():
    def __init__(self, save_name="", cv=5):
        self.save_name = save_name
        self.save_path = save_name + ".pkl"
        self.cv = cv
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define hyperparameter grid
        param_grid = [
        {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000]
        },
        {
            'penalty': ['l1'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['saga', 'liblinear'],
            'max_iter': [500, 1000]
        },
        {
            'penalty': ['elasticnet'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['saga'],
            'max_iter': [500, 1000],
            'l1_ratio': [0.5, 0.7]  # Needed only for elasticnet
        }
    ]

        scorer = make_scorer(f1_score, average='macro')

        # Grid search to find best parameters (not timed)
        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score: {grid_search.best_score_:.4f}")

        # Train best model and measure time
        start_time = time.time()
        self.model = LogisticRegression(random_state=42, **best_params)
        self.model.fit(X_train, y_train)
        end_time = time.time()
        self.time_taken = end_time - start_time

        # Save model and timing info
        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"] 


class Random_Forest():
    def __init__(self, random_state=42, save_name="", cv=5):
        self.random_state = random_state
        self.cv = cv
        self.save_path = save_name + ".pkl"
        self.model = None
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [200, 250, 300, 350],
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 7, 9, 10, 12],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'max_features': ['sqrt', 'log2', None]
        }

        # Use macro F1-score for evaluation
        scorer = make_scorer(f1_score, average='macro')

        # Perform grid search (not timed)
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score from CV: {grid_search.best_score_:.4f}")

        # Time the training of the best model only
        start_time = time.time()
        self.model = RandomForestClassifier(random_state=self.random_state, **best_params)
        self.model.fit(X_train, y_train)
        end_time = time.time()
        self.time_taken = end_time - start_time

        # Save model and time
        model_data = {
            "model": self.model,
            "time": self.time_taken
        }
        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)
    

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]


class Decision_Tree():
    def __init__(self, random_state=42, save_name="", cv=5):
        self.random_state = random_state
        self.save_path = save_name + ".pkl"
        self.cv = cv  # Number of folds for cross-validation
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define hyperparameter search space
        param_grid = {
            'max_depth': [3, 5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['gini', 'entropy', 'log_loss'], 
            'class_weight': [None, 'balanced'],
            'splitter': ['best', 'random']
        }

        # Use F1 Macro as scoring
        scorer = make_scorer(f1_score, average='macro')

        # Grid Search (not timed)
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=self.random_state),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score from CV: {grid_search.best_score_:.4f}")

        # Time training of the best model only
        start_time = time.time()
        self.model = DecisionTreeClassifier(random_state=self.random_state, **best_params)
        self.model.fit(X_train, y_train)
        end_time = time.time()

        self.time_taken = end_time - start_time

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)
    

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]


class Support_Vector_Machine():
    def __init__(self, save_name="", cv=5):
        self.save_path = save_name + ".pkl"
        self.cv = cv
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define parameter grid for SVM
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

        scorer = make_scorer(f1_score, average='macro')

        # Grid search (not timed)
        grid_search = GridSearchCV(
            estimator=SVC(random_state=42, probability=True),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score from CV: {grid_search.best_score_:.4f}")

        # Time training of the best model only
        start_time = time.time()
        self.model = SVC(random_state=42,  probability=True,**best_params)
        self.model.fit(X_train, y_train)
        end_time = time.time()

        self.time_taken = end_time - start_time

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)
    

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]


class XGBoost():
    def __init__(self, random_state=42, save_name="", cv=5):
        self.random_state = random_state
        self.save_path = save_name + ".pkl"
        self.cv = cv
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 75, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.25, 0.3],
            'reg_lambda': [1, 1.5, 2],
            'reg_alpha': [0, 0.5, 1, 1.5, 2]
        }

        # Use F1 Macro as scoring metric
        scorer = make_scorer(f1_score, average='macro')

        # Grid search (not timed)
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score from CV: {grid_search.best_score_}")

        # Time training of the best model only
        start_time = time.time()
        self.model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            **best_params
        )
        self.model.fit(X_train, y_train)
        end_time = time.time()

        self.time_taken = end_time - start_time

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)
    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]


class K_Nearest_Neighbor():
    def __init__(self, save_name="", cv=5):
        self.save_path = save_name + ".pkl"
        self.cv = cv  
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        # Define hyperparameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        # Use F1 Macro as scoring metric
        scorer = make_scorer(f1_score, average='macro')

        # Perform grid search (not timed)
        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score from CV: {grid_search.best_score_}")

        # Now time the training of the best model only
        start_time = time.time()

        self.model = KNeighborsClassifier(**best_params)
        self.model.fit(X_train, y_train)

        end_time = time.time()
        self.time_taken = end_time - start_time

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")
        print(f"Training time for best model: {self.time_taken:.4f} seconds")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
       
        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]


class NavieBayes():
    def __init__(self, save_name = ""):
        self.save_path = save_name + ".pkl"
        self.model = GaussianNB()
    
    def fit_save(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)

        end_time = time.time()

        self.time_taken = end_time - start_time

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data,self.save_path)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate_and_get_metrics(self, X_test, y_test,plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = self.model.predict_proba(X_test)
        # print(y_test.ndim)
        # print("Unique classes in y_test:", np.unique(y_test))
        # print("y_proba shape:", y_proba.shape)
        # print("Sample y_proba rows:", y_proba[:5])

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            time_taken=self.time_taken,
            save_dir=plt_name,
        )

        return metrics


    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]