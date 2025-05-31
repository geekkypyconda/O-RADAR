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
from sklearn.model_selection import TimeSeriesSplit
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
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
# from tensorflow.keras.regularizers import l2 
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,Input
from tensorflow.keras.regularizers import l2
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
        self.threshold = 0.5
        self.time_taken = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if number_of_features is not None:
            self._init_model_()

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
            nn.Linear(32, output_dim)
        )

        if self.num_classes == 2:
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        if self.num_classes == 2:
            y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            y = torch.tensor(y_train, dtype=torch.long).to(self.device)

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
            out = self.forward(X).cpu()

        if self.num_classes == 2:
            out = torch.sigmoid(out)
            preds = (out.numpy() > self.threshold).astype(int)
        else:
            preds = torch.argmax(torch.softmax(out, dim=1), dim=1).numpy()

        return f1_score(y_val, preds, average='macro')

    def fit_save(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            if self.num_classes == 2:
                param_grid = {
                    'learning_rate': [0.01, 0.001],
                    'epochs': [300,350,400,500,700,800,900,1000,1500,2000,3000,4000],
                    'threshold': [0.4, 0.5, 0.6],
                }
            else:
                param_grid = {
                    'learning_rate': [0.01, 0.001],
                    'epochs': [300,350,400,500,700,800,900,1000,1500,2000,3000,4000],
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
                if self.num_classes == 2:
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

        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        if self.num_classes == 2:
            self.threshold = best_params['threshold']
        self._init_model_()

        start_time = time.time()
        self.train_epoch(X_train.to_numpy(), y_train.to_numpy())
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
            logits = self.forward(X_test).cpu()

        if self.num_classes == 2:
            probs = torch.sigmoid(logits).numpy()
        else:
            probs = torch.softmax(logits, dim=1).numpy()

        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        if self.num_classes == 2:
            return (probs > self.threshold).astype(int)
        else:
            return np.argmax(probs, axis=1)

    def evaluate_and_get_metrics(self, X_test, y_test, plt_name):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        proba = self.predict_proba(X_test)

        if self.num_classes == 2:
            y_proba = np.hstack([proba, 1 - proba])
        else:
            y_proba = proba

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
        model_data = torch.load(model_path)
        self.input_dimension = model_data["input_dim"]
        self.num_classes = model_data.get("num_classes", 2)
        self.time_taken = model_data["time"]
        self._init_model_()
        self.model.load_state_dict(model_data["model"])
        self.model.to(self.device)


class Simple_LSTM(nn.Module):
    def __init__(self, number_of_features, timesteps, num_classes=2, learning_rate=0.001,
                 epochs=100, batch_size=32, save_name="lstm_model", cv=5):
        super(Simple_LSTM, self).__init__()
        self.number_of_features = number_of_features
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cv = cv
        self.threshold = 0.5
        self.save_path = save_name + ".pth"
        self.time_taken = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_model_()

    def _init_model_(self):
        output_dim = 1 if self.num_classes == 2 else self.num_classes

        self.lstm1 = nn.LSTM(input_size=self.number_of_features, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, output_dim)

        if self.num_classes == 2:
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):
        # Ensure input is 3D: (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, _ = self.lstm1(x)          # x: (batch, seq_len, 50)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)          # x: (batch, seq_len, 32)
        x = self.dropout2(x)
        x = x[:, -1, :]               # Take last timestep output for classification
        x = self.fc(x)
        return x

    def train_epoch(self, X_train, y_train):
        self.train()
        if hasattr(X_train, "to_numpy"):
            X_np = X_train.to_numpy()
        else:
            X_np = X_train

        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension if missing

        if hasattr(y_train, "to_numpy"):
            y_np = y_train.to_numpy()
        else:
            y_np = y_train

        if self.num_classes == 2:
            y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            y_tensor = torch.tensor(y_np, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.forward(batch_X)
                loss = self.loss_function(output, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, X_val, y_val):
        self.eval()
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        with torch.no_grad():
            outputs = self.forward(X_tensor).cpu()

        if self.num_classes == 2:
            probs = torch.sigmoid(outputs).numpy()
            preds = (probs > self.threshold).astype(int)
        else:
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).numpy()

        return f1_score(y_val, preds, average='macro')

    def fit_save(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            if self.num_classes == 2:
                param_grid = {
                    'learning_rate': [0.001],#, 0.0005],
                    'epochs': [100],# ,150],
                    'threshold': [0.4]#, 0.5, 0.6]
                }
            else:
                param_grid = {
                    'learning_rate': [0.001],# 0.0005],
                    'epochs': [100]#, 150]
                }

        best_score = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            f1_scores = []

            tscv = TimeSeriesSplit(n_splits=self.cv)
            X_train_np = np.array(X_train)
            y_train_np = np.array(y_train)
            for train_idx, val_idx in tscv.split(X_train_np):
                X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
                y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]

                self.learning_rate = params['learning_rate']
                self.epochs = params['epochs']
                if self.num_classes == 2:
                    self.threshold = params['threshold']
                self._init_model_()

                self.train_epoch(X_tr, y_tr)
                f1 = self.evaluate(X_val, y_val)
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = params

        print(f"Best parameters: {best_params}")
        print(f"Best macro F1-score (CV): {best_score:.4f}")

        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        if self.num_classes == 2:
            self.threshold = best_params['threshold']
        self._init_model_()

        start_time = time.time()
        self.train_epoch(X_train, y_train)
        self.time_taken = time.time() - start_time

        torch.save({
            'model_state': self.state_dict(),
            'input_dim': self.number_of_features,
            'timesteps': self.timesteps,
            'num_classes': self.num_classes,
            'time': self.time_taken
        }, self.save_path)

        print(f"Training time: {self.time_taken:.2f} sec")
        print(f"Saved model to {self.save_path}")

    def predict_proba(self, X_test):
        self.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        if X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)
        with torch.no_grad():
            logits = self.forward(X_tensor).cpu()

        if self.num_classes == 2:
            probs = torch.sigmoid(logits).numpy()
        else:
            probs = torch.softmax(logits, dim=1).numpy()

        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        if self.num_classes == 2:
            return (probs > self.threshold).astype(int)
        else:
            return np.argmax(probs, axis=1)

    def evaluate_and_get_metrics(self, X_test, y_test, plt_name=None):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        proba = self.predict_proba(X_test)

        if self.num_classes == 2:
            y_proba = np.hstack([proba, 1 - proba])
        else:
            y_proba = proba

        print(f"Accuracy: {accuracy:.4f}")
        return {
            "accuracy": accuracy,
            "predictions": y_pred,
            "probabilities": y_proba,
            "time_taken": self.time_taken
        }

    def evaluation_mode(self, model_path):
        model_data = torch.load(model_path)
        self.number_of_features = model_data['input_dim']
        self.timesteps = model_data['timesteps']
        self.num_classes = model_data['num_classes']
        self.time_taken = model_data['time']
        self._init_model_()
        self.load_state_dict(model_data['model_state'])
        self.to(self.device)





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

        if(number_of_classes == 2): 
            self.net = nn.Sequential(
                nn.Linear(encoded_dimension, 128),  
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(encoded_dimension, 128),  
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, number_of_classes),
            )

    def forward(self, x):
        return self.net(x)


class Autoencoder_Classifier():
    def __init__(self, X_train = -1, y_train = -1, input_dimension = -1, encoded_dimension = -1,number_of_classes = -1, save_name = ""):
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
            print(f"Tensor: {self.y_train_tensor}")


    def to_tensor(self, X,isLabel):
        if isLabel == True:
            if self.number_of_classes == 2:
                X = torch.tensor(X.to_numpy(), dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                X = torch.tensor(X.to_numpy(), dtype=torch.long).view(-1).to(self.device)
        else:
            X = torch.tensor(X.to_numpy(), dtype=torch.float32).to(self.device)

        return X

    def train_autoencoder(self, epochs = 100, learning_rate = 0.001):
        print("\n\n--------------Training Autoencoder--------\n\n")

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

    def train_classifier(self, epochs = 100, learning_rate=0.01):
        print("\n\n--------------Training Classifier--------\n\n")

        

        if(self.number_of_classes == 2):
            self.clf_criterion = nn.BCEWithLogitsLoss()
        else:
            self.clf_criterion = nn.CrossEntropyLoss()

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


    def evaluate_and_get_metrics(self, X_test, y_test, plt_name):
        X_test_tensor = self.to_tensor(X=X_test, isLabel=False)
        y_test_tensor = self.to_tensor(X=y_test, isLabel=True)

        y_probs = None
        y_pred_tensor = None

        with torch.no_grad():
            encoded_output = self.autoencoder.encoder(X_test_tensor)

            if(self.number_of_classes == 2):
                probs_1 = self.classifier(encoded_output).view(-1,1)
                probs_0 = 1 - probs_1
                y_probs = torch.cat([probs_0, probs_1], dim=1) 
                y_pred_tensor = (probs_1 > 0.5).int()  
            else:
                y_probs = self.classifier(encoded_output)
                y_pred_tensor = torch.argmax(y_probs, dim=1)

        y_pred = y_pred_tensor.cpu().numpy().astype(int)
        y_probs = y_probs.cpu().numpy().astype(float)

        print(f"probs: {y_probs}")

        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_pred_tensor.cpu().numpy())

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken, save_dir=plt_name, y_proba=y_probs)

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
    def __init__(self, n_steps=6, n_d=32, n_a=32,save_name = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
    
        self.save_path = save_name + ".pkl"

        if save_name != "":
            self.clf = TabNetClassifier(
                device_name=self.device.type,
                n_d=n_d,n_a=n_a,
                n_steps=n_steps,
                gamma=1.6,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-4),
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
            eval_metric=['auc'],
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

    def evaluate_and_get_metrics(self, X_test, y_test, plt_name):
        y_pred = self.predict(X_test.values)
        y_probs = self.clf.predict_proba(X_test.values)

        acc = accuracy_score(y_test, y_pred)

        # Defining Metrics for this model
        self.metrics = Metric(accuracy=acc, y_test=y_test, y_pred=y_pred,time_taken=self.time_taken,save_dir=plt_name, y_proba=y_probs)

        return self.metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.clf = model_data["model_object"]

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

    def predict_proba(self,X_test):
        scores = self.model.decision_function(X=X_test)

        outlier_probs = 1 / (1 + np.exp(scores))
        inlier_probs = 1 - outlier_probs

        return np.column_stack((outlier_probs, inlier_probs))

    def evaluate_and_get_metrics(self, X_test, y_test, plt_name):
        y_pred = self.predict(X_test=X_test)
        y_probs = self.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = Metric(accuracy=accuracy, y_test=y_test,y_pred=y_pred,time_taken=self.time_taken, save_dir=plt_name, y_proba=y_probs)

        return metrics

    def evaluation_mode(self, model_path):
        model_data = jlb.load(model_path)
        self.model = model_data["model"]
        self.time_taken = model_data["time"]