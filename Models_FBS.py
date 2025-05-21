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
    recall_score, confusion_matrix, log_loss
)

from sklearn.linear_model import(
    LogisticRegression
)

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ORAN_Helper import Metric
import joblib as jlb

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

class MLP(nn.Module):
    def __init__(self, number_of_features = None, learning_rate=0.001, epochs=100, save_name = ""):
        super(MLP, self).__init__()

        self.input_dimension = number_of_features
        self.epochs = epochs
        self.save_path = save_name + ".pth"
        self.learning_rate =learning_rate

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
        # self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")

    def forward(self,x):
        return self.model(x)

    def fit_save(self,X_train, y_train):
        epochs = self.epochs
        start_time = time.time()
        print_num = epochs // 10

        X = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1).to(self.device)

        for epoch in range(1, epochs + 1):
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.loss_function(out,y)
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

        model_data = {
            "model": self.state_dict(),
            "input_dim": self.input_dimension,
            "time": self.time_taken
        }
        
        torch.save(model_data, self.save_path)

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

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken)

        return self.metrics

    def evaluation_mode(self, model_path):
        model_data = torch.load(model_path)

        self.input_dimension = model_data["input_dim"]
        self.time_taken = model_data["time"]

        self._init_model_()

        self.load_state_dict(model_data["model"])
        self.to(self.device)
    

class Simple_LSTM():
    def __init__(self, timesteps = None, number_of_features = None, learning_rate=0.01, epochs = 100, batch_size = 32, save_name = ""):
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
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.timesteps, self.number_of_features)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid') 
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def transform_label(self, labels,sample_set_size):
        y_seq = (labels.to_numpy().reshape(sample_set_size, self.timesteps).mean(axis=1) >= 0.5).astype(int)

        return y_seq

    def fit_save(self,X_train, y_train):
        sample_set_size = X_train.shape[0] // self.timesteps

        X_train = X_train[:sample_set_size * self.timesteps]
        y_train = y_train[:sample_set_size * self.timesteps]

        y_train = self.transform_label(labels=y_train, sample_set_size=sample_set_size)

        X_train = X_train.to_numpy().reshape((sample_set_size, self.timesteps, self.number_of_features))
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
        probs = self.model.predict_proba(X_test)
        return (probs > 0.5).astype(int)

    def evaluate_and_get_metrics(self, X_test, y_test):
        sample_set_size = X_test.shape[0] // self.timesteps
        X_test = X_test[:sample_set_size * self.timesteps]
        y_test = y_test[:sample_set_size * self.timesteps]

        y_test = self.transform_label(labels = y_test, sample_set_size=sample_set_size)

        X_test = X_test.to_numpy().reshape((sample_set_size, self.timesteps, self.number_of_features))
        y_pred = self.model.predict(X_test)
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


class LR():
    def __init__(self, save_name=""):
        self.save_name = save_name
        self.save_path = save_name + ".pkl"
        self.model = LogisticRegression(random_state=42, max_iter=100)
    
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
        model_data = torch.load(model_path)
        self
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
    def __init__(self, random_state = 42, save_name = ""):
        self.model = DecisionTreeClassifier(random_state=random_state)

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


class Support_Vector_Machine():
    def __init__(self, kernel = "sigmoid", random_state=42, save_name = ""):
        self.model = SVC(kernel=kernel, random_state=random_state)
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
    

class XGBoost():
    def __init__(self,number_of_trees=100, learning_rate=0.1, random_state = 42, save_name = ""):
        self.save_path = save_name + ".pkl"

        self.model = XGBClassifier(n_estimators=number_of_trees, learning_rate=learning_rate,random_state=random_state)
    
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
    
class K_Nearest_Neighbor():
    def __init__(self,neighbors = 5, save_name = ""):
        self.save_path = save_name + ".pkl"

        self.model = KNeighborsClassifier(n_neighbors=neighbors)
    
    def fit_save(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)

        end_time = time.time()

        self.time_taken = end_time - start_time

        jlb.dump(self.model,self.save_path)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

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

    def evaluate_and_get_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test=X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken)

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]
