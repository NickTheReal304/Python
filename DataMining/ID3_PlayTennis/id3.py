"""Docstring checker from the basic checker."""
import pandas as pd
import numpy as np  # for mathematical calculation
# import numpy as np
# importing the dataset from the disk
train_data_m = pd.read_csv("DataMining/ID3_PlayTennis/train/PlayTennis.csv")
# viewing some row of the dataset
train_data_m.head()
print(train_data_m.head())

# Cal the entropy


def calc_total_entropy(train_data, label, class_list):
    total_rows = train_data.shape[0]
    total_entr = 0

    # for each class in the label
    for c in class_list:
        # number of the class
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = \
            - (total_class_count/total_rows) * \
            np.log2(total_class_count/total_rows)  # entropy of the class
        # adding the class entropy to the total entropy of the dataset
        total_entr += total_class_entr

    return total_entr


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        # row count of class c
        label_class_count = \
            feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count  # probability of the class
            entropy_class = - probability_class * \
                np.log2(probability_class)  # entropy
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    # unqiue values of the feature
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        # filtering rows with that feature_value
        feature_value_data = train_data[train_data[feature_name]
                                        == feature_value]
        feature_value_count = feature_value_data.shape[0]
        # calculcating entropy for the feature value
        feature_value_entropy = calc_entropy(
            feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        # calculating information of the feature value
        feature_info += feature_value_probability * feature_value_entropy

    # calculating information gain by subtracting
    return calc_total_entropy(train_data, label, class_list) - feature_info


def find_most_informative_feature(train_data, label, class_list):
    # finding the feature names in the dataset
    feature_list = train_data.columns.drop(label)
    # N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:  # for each feature in the dataset
        feature_info_gain = calc_info_gain(
            feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature
