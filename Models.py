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
    def __init__(self, number_of_features=None, learning_rate=0.001, epochs=100, save_name="", cv=5):
        super(MLP, self).__init__()

        self.input_dimension = number_of_features
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.loss_function(out, y)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_val, y_val):
        X = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.model.eval()
            out = self.forward(X).cpu().numpy()

        if self.loss_function_type == 'bcewithlogits':
            out = torch.sigmoid(torch.tensor(out)).numpy()

        preds = (out > self.threshold).astype(int)
        return f1_score(y_val, preds, average='macro')

    def fit_save(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.001,0.0005],
                'epochs': [300,350,400,500,700,800,900,1000,1500,2000,3000,4000],
                'loss_function_type': ['bce','bcewithlogits'],
                'threshold': [0.4, 0.5, 0.6]
            }

        best_score = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            f1_scores = []

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                self.learning_rate = params['learning_rate']
                self.epochs = params['epochs']
                self.loss_function_type = params['loss_function_type']
                self.threshold = params['threshold']
                self._init_model_()

                self.train_epoch(X_tr.to_numpy(), y_tr.to_numpy())
                f1 = self.evaluate(X_val.to_numpy(), y_val.to_numpy())
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)

            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = params

        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score (CV): {best_score:.4f}")

        # Final training
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        self.loss_function_type = best_params['loss_function_type']
        self.threshold = best_params['threshold']
        self._init_model_()

        start_time = time.time()
        self.train_epoch(X_train.to_numpy(), y_train.to_numpy())
        end_time = time.time()
        self.time_taken = end_time - start_time

        torch.save({
            "model": self.model.state_dict(),
            "input_dim": self.input_dimension,
            "time": self.time_taken
        }, self.save_path)

        print(f"Training time for best model: {self.time_taken:.4f} seconds")
        print(f"Saved best model to {self.save_path}")

    def predict_proba(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.forward(X_test).cpu().numpy()

        if self.loss_function_type == 'bcewithlogits':
            probs = torch.sigmoid(torch.tensor(probs)).numpy()


        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs > self.threshold).astype(int)

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
        self.model.load_state_dict(model_data["model"])
        self.model.to(self.device)


class Simple_LSTM():
    def __init__(self, timesteps = 1, number_of_features = None, learning_rate=0.01, epochs = 100, batch_size = 32, save_name = ""):
        self.save_path = save_name + ".h5"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.number_of_features = number_of_features
        self.timesteps = timesteps

        self.batch_size = batch_size

        if save_name == "":
            pass
        else:
            self._init_model_()

    def _init_model_(self):
        self.model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(self.timesteps, self.number_of_features)),
            BatchNormalization(),
            Dropout(0.4),  # Dropout to prevent overfitting

            LSTM(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),  # Dropout after second LSTM layer

            Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def transform_label(self, labels,sample_set_size):
        y_seq = (labels.to_numpy().reshape(sample_set_size, self.timesteps).mean(axis=1) >= 0.5).astype(int)

        return y_seq

    def fit_save(self,X_train, y_train):
        X_train = X_train.to_numpy().reshape((X_train.shape[0],self.timesteps,X_train.shape[1]))        

        start_time = time.time()

        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

        end_time = time.time()

        self.time_taken = end_time - start_time
        
        self.model.save(self.save_path)
        
        with h5py.File(self.save_path, "a") as file:
            file.attrs["time"] = self.time_taken
            file.attrs["timesteps"] = self.timesteps
            file.attrs["num_features"] = self.number_of_features

    def predict_proba(self, X_test):
        return self.model.predict(X_test)

    def predict(self, X_test):
        probs = self.model.predict(X_test)
        return (probs > 0.5).astype(int)

    def evaluate_and_get_metrics(self, X_test, y_test):
        X_test = X_test.to_numpy().reshape((X_test.shape[0],self.timesteps,X_test.shape[1]))
        y_pred = self.predict(X_test)

        loss, acc = self.model.evaluate(X_test,y_test)

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken)

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

        self.loaded_autoencoder = Autoencoder

        self.input_dimension = model_data["input_dim"]
        self.encoded_dimension = model_data["encoded_dimension"]
        self.time_taken = model_data["time"]
        self.number_of_classes = model_data["num_classes"]

        self.autoencoder = Autoencoder(self.input_dimension,self.encoded_dimension).to(self.device)
        self.autoencoder.load_state_dict(model_data["autoencoder"])
        self.autoencoder.eval()

        self.classifier = Classifier(encoded_dimension=self.encoded_dimension, number_of_classes=self.number_of_classes).to(self.device)
        self.classifier.load_state_dict(model_data["classifier"])
        self.classifier.eval()


class TabNet_Classifier():
    def __init__(self, n_steps=10, n_d=16, n_a=16,save_name = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
    
        self.save_path = save_name + ".pkl"

        if save_name != "":
            self.clf = TabNetClassifier(
                device_name=self.device.type,
                n_d=n_d,n_a=n_a,
                n_steps=n_steps,
                gamma=1.5,
                lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":10, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=1
            )

    def fit_save(self, X_train,y_train,X_test,y_test, epochs,early_stopping_threshold):
        start_time = time.time()

        self.clf.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_test.values, y_test.values)],
            eval_name=['test'],
            eval_metric=['accuracy'],
            max_epochs=epochs,
            patience=early_stopping_threshold,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0
        )

        end_time = time.time()
        time_taken = end_time - start_time

        model_data = {
            "model_object": self.clf,
            "params": self.clf.get_params(),
            "time": time_taken
        }

        jlb.dump(model_data,self.save_path)

    def predict(self,X):
        return self.clf.predict(X)

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test.values)
        acc = accuracy_score(y_test, y_pred)

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken)

        return self.metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.clf = model_data["model_object"]

        self.time_taken = model_data["time"]


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
        # Define parameter grid for SVM
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

        scorer = make_scorer(f1_score, average='macro')

        # Grid search (not timed)
        grid_search = GridSearchCV(
            estimator=SVC(random_state=42),
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
        self.model = SVC(random_state=42, **best_params)
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