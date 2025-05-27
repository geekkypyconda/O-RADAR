import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import importlib
import os
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer,label_binarize
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    log_loss
)


import torch
import torch.nn as nn
import torch.nn.functional as F



class Processor():
    def __init__(self):
        pass

    def get_correlated_features(self, corr, window=0, Threshold = 0.5, print_it = False):
        high_corr_pairs = []
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                # Ensure col1 is not compared to itself, and nearby sensors are excluded
                if col1 != col2 and abs(i - j) > window and abs(corr.loc[col1, col2]) > Threshold:
                    high_corr_pairs.append((col1, col2))
        
        print(f"Number of Highly Co-related feature pairs: {len(high_corr_pairs)} \n\n")

        if print_it:
            for i in high_corr_pairs:
                print(i)

        return high_corr_pairs

    def drop_correlated_features(self, high_corr_pairs, data, print_it = False):
        drop_set = set()

        for pair in high_corr_pairs:
            f1 = pair[0];f2 = pair[1]

            if f1 in drop_set or f2 in drop_set:
                continue

            drop_set.add(f1)

        print(f"Number of features to be dropped: {len(drop_set)}")

        if print_it == True:
            print(drop_set)

        drop_list = list(drop_set)

        data = data.drop(columns=drop_list)

        return data
    
    def get_features_corr_with_label(self, data, label_name = "Label", threshold = 0.5):
        corr_label_fbs_nas = data.corr()["label"].abs()
        relevant_features = corr_label_fbs_nas[corr_label_fbs_nas > threshold]

        # Adding these features into a List
        l = []

        for i in relevant_features.index:
            if i != label_name:
                l.append(i)

        return l

    def get_unique_values_for_each_feature(self, data, print_it = False):
        indent = '{:<3} {:<30}: {}'
        print("Unique Value cout for:")
        unique_dict = {}

        for i, cols in enumerate(list(data.columns), start=1):
            if print_it:
                print(indent.format(f'{i}', cols, data[cols].nunique()))
            unique_dict[cols] = data[cols].nunique()

        return unique_dict

    def remove_features_with_unique_count(self, data,unique_dict, unique_count = 1, remove = False):
        # Iterate throught the dict
        l = []
        for feature in unique_dict.keys():
            if unique_dict[feature] == unique_count:
                l.append(feature)

        if(remove == True):
            new_data = data.drop(columns = l)
            return new_data, l

        return l


    def get_correlatd_count(self, corr_pairs):
        cnt = {}
        for pair in corr_pairs:
            # check if it exists or not
            if pair[0] in cnt:
                cnt[pair[0]] = cnt[pair[0]] + 1
            else:
                cnt[pair[0]] = 1

        return cnt

    def separate_label(self, data, label_name = "Label"):
        labels = data[label_name]
        data_without_labels = data.drop(label_name, axis=1)

        return data_without_labels, labels

    def features_with_zero_std(self, data):
        std = data.std(numeric_only=True)
        zero_std_cols = std[std==0].index.tolist()

        return zero_std_cols

    def create_dataset(self, list_of_selected_features, data, separate_labels = False):
        selected_data = data[list_of_selected_features]

        if separate_labels == True:
            return self.separate_label(selected_data, True)
        else:
            return selected_data

    def scaler(self, data_without_labels, type = "min_max"):
        if type == 'min_max':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(data_without_labels)
            return X
        elif type == "standard":
            scaler = StandardScaler()
            X = scaler.fit_transform(data_without_labels)
            return X
        elif type == "quantile":
            scaler = QuantileTransformer()
            X = scaler.fit_transform(data_without_labels)
            return X
        else:
            print("Wrong Type of Scaler!")
            print("Aborting Operation")
            return None

    def one_hot_encoder(self, labels):
        unique_labels = sorted(labels.unique())
        print(unique_labels)
        # Create a dictionary mapping labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        index_to_label = {i: label for i, label in enumerate(unique_labels)}

        # One-hot encode manually
        y_onehot = np.zeros((len(labels), len(unique_labels)))
        for i, label in enumerate(labels):
            y_onehot[i, label_to_index[label]] = 1
        
        return y_onehot, unique_labels, label_to_index, index_to_label

    def one_hot_decoder(self, label_to_index, index_to_label, one_hot_matrix):
        l = []
        for i in one_hot_matrix:
            idx = np.argmax(i)

            l.append(index_to_label[idx])
            

        return np.array(l)

class Plotter():
    def __init__(self,plt,sns):
        self.plt = plt
        self.sns = sns

        self.good_colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
            "#8c564b",  # chestnut brown
            "#e377c2",  # raspberry yogurt pink
            "#7f7f7f",  # middle gray
            "#bcbd22",  # curry yellow-green
            "#17becf",  # blue-teal
            "#393b79",  # dark muted blue
            "#637939",  # dark olive green
            "#8c6d31",  # dark gold
            "#843c39",  # dark brick red
            "#7b4173",  # dark violet
            "#5254a3",  # blue-violet
            "#9c9ede",  # light blue-violet
            "#6b6ecf",  # moderate blue
            "#b5cf6b",  # light olive green
            "#cedb9c",  # pale olive green
            "#e7ba52",  # sandy brown
            "#e7969c",  # pale red
            "#a55194",  # muted magenta
            "#bd9e39",  # muted gold
            "#ad494a",  # muted red
            "#a55194",  # muted purple
            "#393b79",  # dark blue
            "#8ca252",  # muted green
            "#bca136",  # muted yellow
            "#7b4173",  # deep violet
        ]


    def bar_plot(self,data,x_feature_name, y_feature_name, fig_dimx=10,fig_dimy=8, title="Count Plot", x_label=" X axis", y_label = "Y axis"):
        self.plt.figure(figsize=(fig_dimx,fig_dimy))
        ax = self.sns.barplot(
            x=x_feature_name,
            y=y_feature_name,
            data=data,
            palette='pastel',
            legend=False
        )

        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        self.plt.xticks(rotation = 90)

        for p in self.plt.gca().patches:
            self.plt.gca().annotate(f'{p.get_height():.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                ha='center', fontsize=10)

        self.plt.show()

    def plot_dict(self, dictionary, type="bar", fig_dimx = 10, fig_dimy = 10, title="Count", x_label=" X axis", y_label = "Y axis"):
        keys = list(dictionary.keys())
        values = list(dictionary.values())

        # Generate random colors for each bar
        colors = self.good_colors[:len(keys)]

        self.plt.figure(figsize=(fig_dimx,fig_dimy))

        if type == "bar":
            self.plt.bar(keys, values, color=colors)

            for i, value in enumerate(values):
                self.plt.text(keys[i], value + value * 0.02, str(value), ha='center', fontsize=10)
            
            plt.xticks(keys)
        elif type == "pie":
            x_label = ""
            y_label = ""
            plt.pie(values, labels=keys, colors=colors,autopct='%1.1f%%')
        else:
            print("Invalid Type of plot!")
            return

        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        plt.tight_layout()
        plt.legend()
        self.plt.show()

    def show_plot_for_corr_pairs(self, corr_pairs, fig_dimx = 10, fig_dimy = 10, title="Count", x_label=" X axis", y_label = "Y axis"):
        processor = Processor()
        corr_dict = processor.get_correlatd_count(corr_pairs=corr_pairs)

        self.plot_dict(corr_dict, fig_dimx=fig_dimx, fig_dimy=fig_dimy,title=title,x_label=x_label,y_label=y_label)

    def plot_Train_validation_curves(self, history):
        plt.figure(figsize=(10,5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs. Validation Loss Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training vs. Validation Accuracy Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self,cf_matrix, total_labels, test_labels):
        unique_labels = sorted(total_labels.unique())

        print(f"Number of Test labels: {len(test_labels)}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def plot_grouped_bar_chart(self, datasets, models, mlp_metrics, lstm_metrics,title, y_label, left_margin = 0.001, right_margin = 0.001):
        x = np.arange(len(datasets))
        width = 0.20

        fig, ax = plt.subplots()

        bar1 = ax.bar(x - width/2, mlp_metrics, width, label = "MLP")
        bar2 = ax.bar(x + width/2, lstm_metrics, width, label = "LSTM")

        for bar in bar1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2 - left_margin, height),
                        xytext=(0, 2),  
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bar2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2 + right_margin, height),
                        xytext=(0, 2),  
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_ylabel(f"{y_label} --->")
        ax.set_xlabel("Datasets --->")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        # ax.legend()
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.show()


    def plot_class_wise_barchart(self, datasets, class_f1_scores,title = "",y_label = "" , model_name = ""):
        """
        Parameters:
        - datasets: List of dataset names (e.g., ['Full', 'Reduced'])
        - class_f1_scores: List of lists containing F1 scores for each class
        Format: [[dataset1_class0, dataset1_class1, dataset1_class3], ...]
        - classes: List of original class labels (default [0, 1, 3])
        """

        classes = [0,1,3]
        title= f"{title}: {model_name}"
        y_label=y_label
        figsize=(10, 6)
        bar_width=0.25

        x = np.arange(len(datasets))
        fig, ax = plt.subplots(figsize=figsize)

        # Create bars for each class
        for i, cls in enumerate(classes):
            offsets = bar_width * (i - len(classes)/2 + 0.5)
            scores = [scores[i] for scores in class_f1_scores]
            bars = ax.bar(x + offsets, scores, bar_width, label=f'Class {cls}')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar_width/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_ylabel(f"{y_label} --->")
        ax.set_xlabel("Datasets --->")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


class Metric():
<<<<<<< HEAD
    def __init__(self, accuracy, y_test, y_pred, time_taken, num_labels):
        self.accuracy = accuracy
        self.time_taken = time_taken
        self.f1 = f1_score(y_test, y_pred, average="binary" if num_labels == 2 else "macro")
        self.precision = precision_score(y_test, y_pred, average="binary" if num_labels == 2 else "macro")
        self.recall = recall_score(y_test, y_pred, average="binary" if num_labels == 2 else "macro")
=======
    def __init__(self, accuracy, y_test, y_pred, time_taken, y_proba=None,save_dir=None):
        self.accuracy = accuracy
        self.time_taken = time_taken
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        self.precision = precision_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
>>>>>>> ML_models
        self.cf_matrix = confusion_matrix(y_test, y_pred)
        self.macro_f1 = f1_score(y_test, y_pred, average='macro')
        self.save_dir=save_dir
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_proba = y_proba

        # Try to compute ROC AUC and PR AUC
        try:
            classes = np.unique(y_test)
            if len(classes) == 2 and y_proba.shape[1] == 2:
                # Binary classification case, use probs for positive class only
                self.roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                self.pr_auc = average_precision_score(y_test, y_proba[:, 1])
            else:
                # Multi-class case
                self.y_test_bin = label_binarize(y_test, classes=classes)
                if self.y_test_bin.shape[1] == 1:
                    self.roc_auc = self.pr_auc = None
                else:
                    self.roc_auc = roc_auc_score(self.y_test_bin, y_proba, average="macro", multi_class="ovr")
                    self.pr_auc = average_precision_score(self.y_test_bin, y_proba, average="macro")
        except Exception as e:
            print(f"Warning: Could not compute ROC/PR AUC: {e}")
            self.roc_auc = None
            self.pr_auc = None

    def print_metrics(self):
        print("\nOverall Metrics: \n")
        print(f"--Test Accuracy: {self.accuracy * 100:.2f}%")
        print(f"--Macro F1-Score: {self.macro_f1:.4f}")

        print("\nClass-wise Metrics (weighted):")
        print(f"--Precision: {self.precision:.4f}")
        print(f"--Recall: {self.recall:.4f}")
        print(f"--F1-Score: {self.f1:.4f}")

        print()
        print(f"--ROC AUC: {self.roc_auc if self.roc_auc is not None else 'N/A'}")
        print(f"--PR AUC: {self.pr_auc if self.pr_auc is not None else 'N/A'}")

        print(f"\n\nTime Taken By the Model: {self.time_taken:.4f} seconds\n\n")

    def plot_auc_curves(self):
        if self.roc_auc is None or self.pr_auc is None:
            print("ROC/PR AUC not available for plotting.")
            return

        classes = np.unique(self.y_test)
        n_classes = len(classes)

        if n_classes == 2:
            # Binary case: plot single ROC and PR curve for positive class
            fpr, tpr, _ = roc_curve(self.y_test, self.y_proba[:, 1])
            plt.figure(figsize=(12,5))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/{self.save_dir}@roc_curve.png')
            plt.show()

            precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba[:, 1])
            plt.figure(figsize=(12,5))
            plt.plot(recall, precision, label=f'PR curve (AP = {self.pr_auc:.2f})')
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/{self.save_dir}@PR_curve.png')
            plt.show()

        else:
            # Multi-class case (your existing code)
            plt.figure(figsize=(12, 5))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(self.y_test_bin[:, i], self.y_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score(self.y_test_bin[:, i], self.y_proba[:, i]):.2f})')
            plt.title("ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/{self.save_dir}@roc_curve.png')
            plt.show()

            plt.figure(figsize=(12, 5))
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(self.y_test_bin[:, i], self.y_proba[:, i])
                plt.plot(recall, precision, label=f'Class {i} (AP = {average_precision_score(self.y_test_bin[:, i], self.y_proba[:, i]):.2f})')
            plt.title("Precision-Recall Curves")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/{self.save_dir}@PR_curve.png')
            plt.show()
