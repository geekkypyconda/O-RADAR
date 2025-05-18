"""
This file will take the dataset and save the pre processed dataset into the folder
Pre_Processed_Data_set

You need to give the dataset path as arguments while running this file
Example:  python3 Pre_Processor.py <name_of_the_dataset> yes/no <Name of the saved dataset>

Note: If not name for the saved dataset is given then it will take the name from the original
path of the dataset

Pre_Processing Info:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import importlib
import sys
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

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
    drop_dup     = sys.argv[2]
    folder_path  = "Pre_Processed_Data_set"

    # 1) Load
    dataset = pd.read_csv(dataset_path, index_col=0)
    print(f"Dataset Read Complete, Shape of the dataset: {dataset.shape}")

    # 2) Drop duplicates
    if drop_dup.lower() == "yes":
        print("Dropping Duplicate rows!\n")
        dataset.drop_duplicates(inplace=True)
    else:
        print("Not Dropping Duplicates!\n")

    # 3) Sentinel → NaN
    dataset.replace(-1, np.nan, inplace=True)

    # 4) Separate features & label
    data, labels = processor.separate_label(data=dataset, label_name="label")

    # 5) Impute missing
    numeric_cols     = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # 6) Low-variance filter
    vt = VarianceThreshold(threshold=0.01)
    data = pd.DataFrame(vt.fit_transform(data), columns=data.columns[vt.get_support()],index=data.index)

    # 7) Correlation filter
    corr_matrix      = data.corr().round(4)
    high_corr_pairs  = processor.get_correlated_features(corr=corr_matrix,
                                                         Threshold=0.95,
                                                         print_it=False)
    data_processed   = processor.drop_correlated_features(high_corr_pairs=high_corr_pairs,
                                                          data=data)

    # 8) Mutual Information filter
    mi_scores = mutual_info_classif(data_processed, labels)
    mi_series = pd.Series(mi_scores, index=data_processed.columns)
    keep_cols = mi_series[mi_series > 0.01].index
    data_processed = data_processed[keep_cols]

    # 9) Re-attach label
    final_data = data_processed.copy()
    final_data["label"] = labels

    print(f"\nFinal Shape of dataset after feature selection: {final_data.shape}\n")

    # 10) Skipped Scaling

    # 11) Save
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    fn = extract_dataset_name()
    if len(sys.argv) == 4:
        fn = sys.argv[3]
    save_path = os.path.join(folder_path, fn)
    final_data.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
