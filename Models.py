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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("\n\n\n<<<<<<<<<<<<<<----------------------------------->>>>>>>>>>>>>>>>>>>")
print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
print("<<<<<<<<<<<<<<----------------------------------->>>>>>>>>>>>>>>>>>>\n\n\n")



class MLP(nn.Module):
    def __init__(self, number_of_features=None, num_classes=2, learning_rate=0.001, epochs=100, save_name="", cv=5):
        super(MLP, self).__init__()
        self.input_dimension = number_of_features
        self.num_classes = num_classes
        self.epochs = epochs
        self.save_path = save_name + ".pth"
        self.learning_rate = learning_rate
        self.cv = cv
        self.loss_function_type = 'bcewithlogits'  # default
        self.threshold = 0.5
        self.time_taken = None

        if number_of_features is not None:
            self._init_model_()
            print(f"---->>>>>>> Using device: {self.device}")

    def _init_model_(self):
        output_dim = 1 if self.num_classes == 2 else self.num_classes

        self.model = nn.Sequential(
            nn.Linear(self.input_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        if self.loss_function_type == 'bce':
            self.model.add_module('Sigmoid', nn.Sigmoid())
            self.loss_function = nn.BCELoss()
        elif self.loss_function_type == 'bcewithlogits':
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Invalid loss_function_type. Use 'bce' or 'bcewithlogits'.")

        # self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")

    def forward(self, x):
        return self.model(x)

    def fit_save(self,X_train, y_train):
        epochs = self.epochs
        start_time = time.time()
        print_num = epochs // 10

        X = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.loss_function(out, y)
            loss.backward()
            self.optimizer.step()

            preds = (out > 0.5).int().cpu().numpy()
            y_true = y.cpu().numpy()
            accuracy = accuracy_score(y_true, preds)

            if epoch == 1 or epoch % print_num == 0 or epoch == epochs:
                print(f"Epoch {epoch} ---->>>>>>>>>>, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f}")
                print()

        end_time = time.time()
        self.time_taken = end_time - start_time

        torch.save({
            "model": self.model.state_dict(),
            "input_dim": self.input_dimension,
            "num_classes": self.num_classes,
            "time": self.time_taken
        }, self.save_path)

        print(f"Training time for best model: {self.time_taken:.4f} seconds")
        print(f"Saved best model to {self.save_path}")

    def predict_proba(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.forward(X_test).cpu().numpy()
        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs > 0.5).astype(int)

    def evaluate_and_get_metrics(self, X_test, y_test):
        sample_set_size = X_test.shape[0] // self.timesteps
        X_test = X_test[:sample_set_size * self.timesteps]
        y_test = y_test[:sample_set_size * self.timesteps]

        y_test = self.transform_label(labels = y_test, sample_set_size=sample_set_size)

        X_test = X_test.to_numpy().reshape((sample_set_size, self.timesteps, self.number_of_features))
        y_pred = self.model.predict(X_test)
        loss, acc = self.model.evaluate(X_test,y_test)

        self.metrics = Metric(
            accuracy=acc,
            y_test=y_test,
            y_pred=y_pred,
            time_taken=self.time_taken
        )

        return self.metrics

    def evaluation_mode(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

        with h5py.File(model_path, "r") as file:
            self.time_taken = file.attrs["time"]
            self.timesteps = file.attrs["timesteps"]
            self.number_of_features = file.attrs["num_features"]

class Autoencoder(nn.Module):
    def __init__(self, input_dimension, encoded_dimension):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 32),
            nn.ReLU(),
            nn.Linear(32,encoded_dimension)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_dimension,32),
            nn.ReLU(),
            nn.Linear(32, input_dimension)
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    

class Classifier(nn.Module):
    def __init__(self, encoded_dimension,number_of_classes):
        super(Classifier,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(encoded_dimension,16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Autoencoder_Classifier():
    def __init__(self, X_train = -1, y_train = -1, input_dimension = -1, encoded_dimension = -1,number_of_classes = -1, autoencoder_learning_rate=0.01, save_name = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        if(input_dimension != -1):
            self.input_dimension = input_dimension
            self.encoded_dimension = encoded_dimension
            self.number_of_classes = number_of_classes
            self.X_train = X_train

            self.save_path = save_name + ".pth"

            self.autoencoder = Autoencoder(input_dimension=input_dimension,encoded_dimension=encoded_dimension).to(self.device)
            self.classifier = Classifier(encoded_dimension=encoded_dimension,number_of_classes=number_of_classes).to(self.device)
        
            self.X_train_tensor = self.to_tensor(X_train, False)
            self.y_train_tensor = self.to_tensor(y_train, isLabel=True)
       

    def to_tensor(self, X,isLabel):
        if isLabel == True:
            X = torch.tensor(X.to_numpy(), dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            X = torch.tensor(X.to_numpy(), dtype=torch.float32).to(self.device)

        return X

    def train_autoencoder(self, epochs = 100, learning_rate = 0.001):
        self.ae_criterion = nn.MSELoss()
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        print_num = epochs // 10
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.autoencoder.train()
            self.ae_optimizer.zero_grad()
            reconstructed = self.autoencoder(self.X_train_tensor)
            loss = self.ae_criterion(reconstructed,self.X_train_tensor)
            loss.backward()
            self.ae_optimizer.step()

            if epoch == 1 or epoch % print_num == 0 or epoch == epochs:
                print(f"Epoch {epoch} ---->>>>>>>>>>, Loss: {loss.item():.4f}")
                print()
    
        end_time = time.time()

        self.autoencoder_time = end_time - start_time

        # Freeze Autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.encoded_train = self.autoencoder.encoder(self.X_train_tensor).detach()

    def train_classifier(self, epochs = 100, learning_rate=0.001):
        self.clf_criterion = nn.BCEWithLogitsLoss()
        self.clf_optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        print_num = epochs // 10

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.classifier.train()
            self.clf_optimizer.zero_grad()
            outputs = self.classifier(self.encoded_train)
            loss = self.clf_criterion(outputs,self.y_train_tensor)
            loss.backward()
            self.clf_optimizer.step()

            # Calculate training accuracy
            with torch.no_grad():
                predictions_train = outputs.argmax(dim=1)
                train_acc = accuracy_score(self.y_train_tensor.cpu().numpy(), predictions_train.cpu().numpy())

            if epoch == 1 or epoch % print_num == 0 or epoch == epochs:
                print(f"Epoch {epoch} ---->>>>>>>>>>, Loss: {loss.item():.4f}, Training Accuracy: {train_acc}")
                print()

        end_time = time.time()

        self.clf_time = end_time - start_time
        self.time_taken = self.clf_time + self.autoencoder_time

    def save_model(self):
        model_data = {
            "autoencoder": self.autoencoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "time": self.time_taken,
            "input_dim":self.input_dimension,
            "encoded_dimension": self.encoded_dimension,
            "num_classes": self.number_of_classes
        }
        
        torch.save(model_data, self.save_path)


    def evaluate_and_get_metrics(self, X_test, y_test):
        X_test_tensor = self.to_tensor(X=X_test, isLabel=False)
        y_test_tensor = self.to_tensor(X=y_test, isLabel=True)

        with torch.no_grad():
            encoded_output = self.autoencoder.encoder(X_test_tensor)
            y_pred_tensor = (self.classifier(encoded_output) > 0.5).float()

        y_pred = y_pred_tensor.cpu().numpy().astype(int)

        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_pred_tensor.cpu().numpy())

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

        return metrics

    def evaluation_mode(self, model_path):
        model_data = torch.load(model_path)

        self.input_dimension = model_data["input_dim"]
        self.encoded_dimension = model_data["encoded_dimension"]
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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

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





class Support_Vector_Machine():
    def __init__(self, save_name="", cv=5):
        self.save_path = save_name + ".pkl"
        self.cv = cv
        self.time_taken = None

    def fit_save(self, X_train, y_train):
        start_time = time.time()
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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

        return metrics

    def evaluation_mode(self, model_path):
        self.model = jlb.load(model_path)

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