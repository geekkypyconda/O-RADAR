'''
This file will take the dataset and save the pre processed dataset into the folder
Pre_Processed_Data_set

You need to give the dataset path as arguments while running this file
Example:  python3 Pre_Processor.py <path_of_the_dataset> yes/no <Name of the saved dataset>

Note: If not name for the saved dataset is given then it will take the name from the original
path of the dataset

Pre_Processing Info:

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import importlib
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

from ORAN_Helper import *

dataset_path = ""
processor = Processor()
dataset = None

def extract_dataset_name():
    global dataset_path

    file_name = os.path.basename(dataset_path)

    return file_name


def main():
    global dataset_path
    dataset_path = sys.argv[1]
    drop_dup = sys.argv[2]
    folder_path = "Pre_Processed_Data_set"

    dataset = pd.read_csv(dataset_path, index_col=0)
    print(f"Dataset Read Complete, Shape of the dataset: {dataset.shape}")

    # Dropping the duplicate rows
    if(drop_dup == "yes"):
        print("Dropping Duplicate rows!\n")
        dataset.drop_duplicates(inplace=True)
    else:
        print("Not Dropping Duplicates!\n")
    
    # Separating data and labels
    data, labels = processor.separate_label(data=dataset, label_name="label")

    # Dropping Highly Correlated Features
    corr_matrix = data.corr().round(4)
    high_corr_pairs = processor.get_correlated_features(corr=corr_matrix, Threshold=0.95, print_it=False)

    data_processed = processor.drop_correlated_features(high_corr_pairs=high_corr_pairs, data=data)

    final_data = data_processed.copy()
    final_data["label"] = labels

    print(f"\n Final Shape of dataset: {final_data.shape}\n")

    print("Saving the pre-processed dataset!")

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    fn = extract_dataset_name()

    # Now if filename is given then update
    if len(sys.argv) == 4:
        fn = sys.argv[3]

    save_path = folder_path + "/" + fn

    final_data.to_csv(save_path, index = False)


if __name__ == "__main__":
    main()