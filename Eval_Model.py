'''
How to run

python3 Eval_Model.py <Dataset_path> <Saved Model path>

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

from Models import *
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

reverse_models_mapping = {
    "Logistic_Regression": 1,
    "Decision_Tree": 2,
    "Random_Forest": 3,
    "Isolation_Forest": 4,
    "SVM": 5,
    "XGBOOST": 6,
    "K-NN": 7,
    "Naive_Bayes": 8,
    "MLP": 9,
    "LSTM": 10,
    "Stateful_Attention_LSTM": 11,
    "Autoencoder": 12,
    "TabNet": 13
}

def extract_dataset_name(dataset_path):
    file_name = os.path.basename(dataset_path)

    return file_name

def extract_model_number(model_path):
    l = model_path.split('@')[1].split(".")[0]
    
    return reverse_models_mapping[l]

def eval_model(model_num, model_path,X_test,y_test):
    model = None
    if model_num == 1:
        model = LR()
    elif model_num == 2:
        model = Decision_Tree()
    elif model_num == 3:
        model = Random_Forest()
    elif model_num == 4:
        model = Isolation_Forest()
    elif model_num == 5:
        model = Support_Vector_Machine()
    elif model_num == 6:
        model = XGBoost()
    elif model_num == 7:
        model = K_Nearest_Neighbor()
    elif model_num == 8:
        model = NavieBayes()
    elif model_num == 9:
        model = MLP()
    elif model_num == 10:
        model = Simple_LSTM()
    elif model_num == 12:
        model = Autoencoder_Classifier()
    elif model_num == 13:
        model = TabNet_Classifier()
    elif model_num == 11:
        pass
    
    # Put the model in evaluation mode
    model.evaluation_mode(model_path=model_path)

    # get the metrics
    metrics = model.evaluate_and_get_metrics(X_test=X_test, y_test=y_test)

    # print the metrics
    metrics.print_metrics()

def main():
    if len(sys.argv) < 3:
        print("No Model Selected!")
        exit(0)

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    dataset = pd.read_csv(dataset_path)

    data, labels = processor.separate_label(data=dataset, label_name="label")

    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=42)

    print("\n\n-----------------------------------------------")
    print(f"Training Set dimensions: {X_train.shape}")
    print(f"Testing Set dimensions: {X_test.shape}")
    print("-----------------------------------------------\n\n")

    model_num = extract_model_number(model_path=model_path)
    print(f"Evaluating on Model: {models_mapping[model_num]}")
    eval_model(model_num=model_num, model_path=model_path,X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()