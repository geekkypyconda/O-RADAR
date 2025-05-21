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
from ORAN_Helper import Metric
import joblib as jlb
from sklearn.model_selection import GridSearchCV
class MLP(nn.Module):
    def __init__(self, number_of_features=None, learning_rate=0.001, epochs=100, save_name="", cv=5):
        super(MLP, self).__init__()

        self.input_dimension = number_of_features
        self.epochs = epochs
        self.save_path = save_name + ".pth"
        self.learning_rate = learning_rate
        self.cv = cv

        if number_of_features is not None:
            self._init_model_()

    def _init_model_(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.loss_function(out, y)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_val, y_val):
        X = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.eval()
            preds = self.forward(X).cpu().numpy()
        preds = (preds > 0.5).astype(int)
        return f1_score(y_val, preds, average='macro')

    def fit_save(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.001],
                'epochs': [50, 100]
            }

        best_score = -np.inf
        best_params = None
        start_time = time.time()

        for params in ParameterGrid(param_grid):
            f1_scores = []

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                self.learning_rate = params['learning_rate']
                self.epochs = params['epochs']
                self._init_model_()

                self.train_epoch(X_tr.to_numpy(), y_tr.to_numpy())
                f1 = self.evaluate(X_val.to_numpy(), y_val.to_numpy())
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)

            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = params

        # Retrain on all training data with best params
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        self._init_model_()
        self.train_epoch(X_train.to_numpy(), y_train.to_numpy())
        end_time = time.time()
        self.time_taken = end_time - start_time

        # Save model
        torch.save({
            "model": self.state_dict(),
            "input_dim": self.input_dimension,
            "time": self.time_taken
        }, self.save_path)

        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score (CV): {best_score:.4f}")
        print(f"Saved best model to {self.save_path}")

    def predict_proba(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.forward(X_test).cpu().numpy()
        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs > 0.5).astype(int)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.metrics = Metric(
            accuracy=acc,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
        )

        return self.metrics

    def evaluation_mode(self, model_path):
        model_data = torch.load(model_path)
        self.input_dimension = model_data["input_dim"]
        self.time_taken = model_data["time"]
        self._init_model_()
        self.load_state_dict(model_data["model"])
        self.to(self.device)


class LSTM():
    def __init__(self):
        pass



class LR():
    def __init__(self, save_name="", cv=5):
        self.save_name = save_name
        self.save_path = save_name + ".pkl"
        self.cv = cv
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        start_time = time.time()

        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['saga', 'lbfgs', 'liblinear'],
            'max_iter': [100, 200, 300]
        }

        scorer = make_scorer(f1_score, average='macro')

        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        end_time = time.time()
        self.time_taken = end_time - start_time

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best macro F1-score: {grid_search.best_score_}")

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
        )

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]
        
class Isolation_Forest():
    def __init__(self, number_of_trees=100, random_state=42,contamination=0.05, save_name = ""):
        self.model = IsolationForest(n_estimators=number_of_trees,contamination=contamination,random_state=random_state)

        self.save_path = save_name + ".pkl"
    
    def fit_save(self, X_train):
        start_time = time.time()
        self.model.fit(X_train)

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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"] 


class Random_Forest():
    def __init__(self, number_of_trees = 100,random_state = 42, save_name = ""):
        self.number_of_trees = number_of_trees
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=number_of_trees,random_state=random_state)

        self.save_path = save_name + ".pkl"
    
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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

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
        start_time = time.time()

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

        # Perform Grid Search CV
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=self.random_state),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        end_time = time.time()
        self.time_taken = end_time - start_time

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best macro F1-score: {grid_search.best_score_}")

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
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
        start_time = time.time()

        # Define parameter grid for SVM
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

        scorer = make_scorer(f1_score, average='macro')

        grid_search = GridSearchCV(
            estimator=SVC(random_state=42),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        end_time = time.time()
        self.time_taken = end_time - start_time

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best macro F1-score: {grid_search.best_score_:.4f}")

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
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
        start_time = time.time()

        param_grid = {
            'n_estimators': [50,75, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.25, 0.3],
            'reg_lambda': [1, 1.5, 2],
            'reg_alpha': [0, 0.5, 1, 1.5 , 2]
        }

        scorer = make_scorer(f1_score, average='macro')

        grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss'),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        end_time = time.time()
        self.time_taken = end_time - start_time

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best macro F1-score: {grid_search.best_score_}")

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
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
        start_time = time.time()

        # Define hyperparameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7,8, 9,10,11,12,13,14],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        # Use F1 Macro as scoring metric
        scorer = make_scorer(f1_score, average='macro')

        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )

        # Perform the grid search
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        end_time = time.time()
        self.time_taken = end_time - start_time

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best macro F1-score: {grid_search.best_score_}")

        model_data = {
            "model": self.model,
            "time": self.time_taken
        }

        jlb.dump(model_data, self.save_path)
        print(f"Saved best model to {self.save_path}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(
            accuracy=accuracy,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]
