'''
How to run

python3 Train_model.py <Dataset_path> <list of numbers of models>

All the models will be saved in a folder called Saved_Models

dataset@model

Models used:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Isolation Forest
5. SVM
6. XGBOOST
7. K-NN
8. Naive Bayes
9. MLP
10. LSTM
11. Statefull LSTM with Attention
12. Autoencoder
13. TabNet

'''

import sys
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=FitFailedWarning)
# warnings.filterwarnings('ignore', message=r".*Parameters: * are not used.*")

from ML_Models import *
from DL_Models import *
from ORAN_Helper import Processor

save_folder_path = "Saved_Models"
processor = Processor()

models_mapping = {
    1: "Logistic_Regression",
    2: "Decision_Tree",
    3: "Random_Forest",
    4: "Isolation_Forest",
    5: "SVM",
    6: "XGBOOST",
    7: "K-NN",
    8: "Naive_Bayes",
    9: "MLP",
    10: "LSTM",
    11: "Stateful_Attention_LSTM",
    12: "Autoencoder",
    13: "TabNet"
}

def extract_dataset_name(dataset_path):
    file_name = os.path.basename(dataset_path)

    return file_name


def run_model(model_num, dataset_path,X_train,y_train,X_test,y_test):
    model = None
    input_dimension = X_train.shape[1]
    number_of_classes = len(np.unique(y_train))

    save_name = save_folder_path + "/" + extract_dataset_name(dataset_path) + "@" + models_mapping[model_num]
    if model_num == 1:
        model = LR(save_name=save_name)
    elif model_num == 2:
        model = Decision_Tree(random_state=42, save_name=save_name)
    elif model_num == 3:
        model = Random_Forest( save_name=save_name)
    elif model_num == 4:
        model = Isolation_Forest(number_of_trees=100, contamination=0.1, save_name=save_name)
    elif model_num == 5:
        model = Support_Vector_Machine(save_name=save_name)
    elif model_num == 6:
        model = XGBoost( save_name=save_name)
    elif model_num == 7:
        model = K_Nearest_Neighbor(save_name=save_name)
    elif model_num == 8:
        model = NavieBayes(save_name=save_name)
    elif model_num == 9:
        model = MLP(number_of_features=X_train.shape[1],learning_rate=0.001, save_name=save_name)
    elif model_num == 10: # We have to give 1 timestep, because our dataset is like that only
        model = Simple_LSTM(num_classes=number_of_classes, timesteps=1, number_of_features=X_train.shape[1],learning_rate=0.01,epochs=100, batch_size=32,save_name=save_name)
    elif model_num == 12:
        model = Autoencoder_Classifier(X_train=X_train,y_train=y_train,input_dimension=input_dimension, encoded_dimension=24, number_of_classes=number_of_classes, save_name=save_name)
    elif model_num == 13:
        model = TabNet_Classifier(save_name=save_name)
    

    if model_num == 4:
        model.fit_save(X_train=X_train)
    elif model_num == 12:
        model.train_autoencoder(epochs=100,learning_rate=0.01)
        model.train_classifier(epochs=100,learning_rate=0.01)
        model.save_model()
    elif model_num == 13:
        model.fit_save(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,epochs=100,early_stopping_threshold=15)
    else:
        model.fit_save(X_train=X_train,y_train=y_train)


def main():
    if len(sys.argv) < 3:
        print("No Model Selected!")
        exit(0)
    
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    dataset_path = sys.argv[1]
    dataset = pd.read_csv(dataset_path)

    data, labels = processor.separate_label(data=dataset, label_name="label")

    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, stratify=labels, random_state=42)

    # 9) Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_train)
    # data_scaled = pd.DataFrame(data_scaled, columns=data_processed.columns)
    data_scaled = pd.DataFrame(data_scaled, columns=X_train.columns, index=X_train.index)

    folder_path  = "Saved_Models"
    # Save the scaler
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    fn = os.path.splitext(os.path.basename(dataset_path))[0]
    if len(sys.argv) == 4:
        fn = sys.argv[3]
    scaler_path = os.path.join(folder_path, f"{os.path.splitext(fn)[0]}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    #print("Checking X_train for NaNs:", np.isnan(X_train).sum())
    #print("Checking X_train for Infs:", np.isinf(X_train).sum())
    #print("Checking y_train for NaNs:", np.isnan(y_train).sum())
    #print("Checking y_train for Infs:", np.isinf(y_train).sum())

    # Check dtypes and shapes before conversion
   # print("X_train dtype before:", X_train.dtypes)
   # print("y_train dtype before:", y_train.dtypes)
   # print("y_train shape before:", y_train.shape)
    
    l = list(sys.argv[2:])

    print("\n\n-----------------------------------------------")
    print(f"Training Set dimensions: {X_train.shape}")
    print(f"Testing Set dimensions: {X_test.shape}")
    print("-----------------------------------------------\n\n")

    for i in l:
        model_num = int(i)
        print(f"Training on Model: {models_mapping[model_num]}")

        run_model(model_num=model_num,dataset_path=dataset_path,X_train=data_scaled,y_train=y_train,X_test=X_test,y_test=y_test)

        print()


if __name__ == "__main__":
    main()
