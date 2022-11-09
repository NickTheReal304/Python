import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
data2 = pd.read_csv("./Kaggle/dataset/PlayTennis.csv")
data1 = pd.read_csv("./Kaggle/dataset/train.csv")


def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]  # the total size of the dataset
    total_entr = 0

    for c in class_list:  # for each class in the label (0 - 664 )

        # number of the class
        total_class_count = train_data[train_data[label] == c].shape[0]

        total_class_entr = - (total_class_count/total_row) * \
            np.log(total_class_count/total_row) / \
            np.log(class_list.shape[0])  # entropy of the class

        # adding the class entropy to the total entropy of the dataset
        total_entr += total_class_entr

    return total_entr


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        # row count of class c
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count  # probability of the class
            entropy_class = - probability_class * \
                (np.log(probability_class) /
                 np.log(class_list.shape[0]))  # entropy
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
    feature_list = train_data.columns.drop(
        ['Id', label])
    # feature_list = train_data.columns.drop(label)

    # N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:  # for each feature in the dataset
        feature_value_list = train_data[feature].unique()
        # hasNaN = False

        # for feature_class in feature_value_list:
        if pd.isnull(feature_value_list[0]):
            # hasNaN = True
            continue

        print("Has feature: " + feature)
        feature_info_gain = calc_info_gain(
            feature, train_data, label, class_list)
        print(feature_info_gain)
        if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(
        sort=False)  # dictionary of the count of unqiue feature value
    tree = {}  # sub tree or node

    for feature_value, count in feature_value_count_dict.items():
        # dataset with only feature_name = feature_value
        feature_value_data = train_data[train_data[feature_name]
                                        == feature_value]

        assigned_to_node = False  # flag for tracking feature_value is pure class or not
        for c in class_list:  # for each class
            # count of class c
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            # count of (feature_value = count) of class (pure class)
            if class_count == count:
                tree[feature_value] = c  # adding node to the tree
                # removing rows with feature_value
                train_data = train_data[train_data[feature_name]
                                        != feature_value]
                assigned_to_node = True
        if not assigned_to_node:  # not pure class
            # as feature_value is not a pure class, it should be expanded further,
            tree[feature_value] = "?"
            # so the branch is marking with ?
    return tree, train_data


def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:  # if dataset becomes empty after updating
        max_info_feature = find_most_informative_feature(
            train_data, label, class_list)  # most informative feature
        # getting tree node and updated dataset
        tree, train_data = generate_sub_tree(
            max_info_feature, train_data, label, class_list)
        next_root = None

        if prev_feature_value != None:  # add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:  # add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()):  # iterating the tree node
            if branch == "?":  # if it is expandable
                # using the updated dataset
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label,
                          class_list)  # recursive call with updated dataset


def id3(train_data_m, label):
    train_data = train_data_m.copy()  # getting a copy of the dataset
    tree = {}  # tree which will be updated
    # getting unqiue classes of the label
    class_list = train_data[label].unique()
    # start calling recursion
    make_tree(tree, None, train_data, label, class_list)
    return tree


def main():
    # print(calc_total_entropy(data1, 'SalePrice',
    #       data1['SalePrice'].unique()))

    # print(calc_entropy(data1[data1['Street'] == 'Pave'],  'SalePrice',
    #       data1['SalePrice'].unique()))
    # print(find_most_informative_feature(
    # data1, 'SalePrice', data1['SalePrice'].unique()))
    # print(calc_info_gain('LotArea', data1,
    #       'SalePrice', data1['SalePrice'].unique()))
    # tree = id3(data1[:50], 'SalePrice')
    # print(tree)

    # max_info_feature = find_most_informative_feature(
    #     data1, 'SalePrice', data1['SalePrice'].unique())  # most informative feature
    # # getting tree node and updated dataset
    # tree, data2 = generate_sub_tree(
    #     max_info_feature, data1, 'SalePrice', data1['SalePrice'].unique())
    # print("tree: " + str(tree))
    # print(data2)
    tD = data1.copy()  # getting a copy of the dataset
    tree = {}

    make_tree(tree, None, tD, 'SalePrice', data1['SalePrice'].unique())
    print("tree: " + str(tree))


if __name__ == "__main__":
    main()
