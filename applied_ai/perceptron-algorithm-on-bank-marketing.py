'''
The classification goal is to predict if the client will subscribe a term deposit (variable y).

Additional Information

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

There are four datasets: 
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). 
4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). 
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). 

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

Has Missing Values?

No
'''
import os
import pickle
from random import random, randrange

import pandas as pd
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder)
from ucimlrepo import fetch_ucirepo

RAW_DATA_FILE = "../data/interim/bank_marketing.pkl"
# fetch dataset
if os.path.exists(RAW_DATA_FILE):
    with open(RAW_DATA_FILE, 'rb') as f:
        bank_marketing = pickle.load(f)
else:
    bank_marketing = fetch_ucirepo(id=222)
    os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)
    with open(RAW_DATA_FILE, 'wb') as f:
        pickle.dump(bank_marketing, f)


# data (as pandas dataframes)
X = pd.DataFrame(bank_marketing.data.features)
y = pd.Series(bank_marketing.data.targets["y"])

# variable information
print(bank_marketing.variables)


# Replace missing values
def impute_missing_values(X: pd.DataFrame):
    # contact
    categories = X["contact"].unique().tolist()
    X["contact"] = X["contact"].fillna('other')
    categories = X["contact"].unique().tolist()
    print(categories)

    # pdays
    print(X["pdays"].describe())
    MAX_PDAYS = X["pdays"].max()
    print(f"MAX_PDAYS={MAX_PDAYS}")
    pdays = X["pdays"].unique().tolist()
    print(pdays)
    X["pdays"] = X["pdays"].replace(-1, MAX_PDAYS + 1)
    print(X["pdays"].describe())

    # poutcome
    print("Nan Poutcome =", len(X[X["poutcome"].isna() == True]), "Total =", len(X["poutcome"]))
    # Since 80% of poutcomes are nan, it is better to drop this column

    del X["poutcome"]
    print(X.columns)

    # Check which columns still have nan
    print(X.isnull().sum())

    # job
    unique_jobs = X["job"].unique().tolist()
    print("Unique jobs", unique_jobs)
    # Replace nan with other
    X["job"] = X["job"].fillna("other")

    # Check which columns still have nan
    print(X.isnull().sum())

    # education
    educations = X["education"].unique().tolist()
    print("Educations ", educations)
    X["education"] = X["education"].fillna("other")

    # martial
    unique_martial = X["marital"].unique().tolist()
    print("Martial", unique_martial)

    print(X.isnull().sum())
    return X


def replace_categorical_values(X: pd.DataFrame):
    # OneHot encode categorical data
    onehot_encoder = OneHotEncoder(sparse_output=False)
    print("Unique values for contact", X["contact"].unique().tolist())
    print("Unique values for job", X["job"].unique().tolist())
    print("Unique values for education", X["education"].unique().tolist())
    print("Unique values for martial", X["marital"].unique().tolist())
    print("Unique values for default", X["default"].unique().tolist())
    print("Unique values for housing", X["housing"].unique().tolist())
    print("Unique values for loan", X["loan"].unique().tolist())
    print("Unique values for day_of_week", X["day_of_week"].unique().tolist())
    print("Unique values for month", X["month"].unique().tolist())

    categorical_df = X[["job", "marital", "education", "default", "housing", "loan", "contact"]]
    print(categorical_df)

    # ordinal encoding
    ordinal_encoder = OrdinalEncoder(categories=[["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]])
    ordinal_df = X[["month"]]

    print("X before onehot and ordinal encoding", X)
    print(X.columns)
    X = X.drop(columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month"])

    onehot_encoded_df = pd.DataFrame(onehot_encoder.fit_transform(categorical_df))
    ordinal_encoded_df = pd.DataFrame(ordinal_encoder.fit_transform(ordinal_df))
    X = pd.concat([X, onehot_encoded_df, ordinal_encoded_df], ignore_index=True, axis=1,)
    print("X shape after onehot and ordinal encoding", X.shape)
    print("X columns", X.columns)

    print(X)
    return X


def normalize_data(X: pd.DataFrame):
    print(X.describe())

    # normalize the columns whose min != 0 and max!=0
    scaler = MinMaxScaler(feature_range=(0, 1))
    columns_to_be_normalized = []
    print("Index", X.columns)
    for i in X.columns:
        if X.min()[i] != 0.0 or X.max()[i] != 1.0:
            columns_to_be_normalized.append(i)

    X[columns_to_be_normalized] = pd.DataFrame(scaler.fit_transform(X[columns_to_be_normalized]))
    print(X.describe())
    return X


def encode_target_variable(y: pd.Series):

    print("Unique values of y", y.unique())
    print(y.head())
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y))
    print("Shape of y", y.shape)
    print("Unique values of y", y.unique())
    print(y.head())
    return y


def k_fold_cross_validation(dataset: pd.DataFrame, k: int):
    dataset_copy = list(dataset)
    folds = []
    fold_size = len(dataset_copy) // k

    for i in range(k):
        fold = []
        while len(fold) < fold_size:
            idx = randrange(len(dataset_copy))
            fold.append(dataset_copy[idx])
            dataset_copy.pop(idx)

        folds.append(fold)

    return folds


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return (correct / len(actual)) * 100


def predict(row, weights):
    activation = weights[0]

    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]

    return 1.0 if activation >= 0 else 0.0


def train_weights(train_set, lr_rate, n_iter):
    weights = [random() for i in range(len(train_set[0]))]

    for epoch in range(n_iter):
        error_sum = 0
        for row in train_set:
            predicted = predict(row, weights)
            expected = row[-1]
            error = predicted - expected
            error_sum += error**2
            # Update weights
            weights[0] = weights[0] + lr_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + lr_rate * error + row[i]
        print(">epoch=%d lr_rate=%.2f error=%.2f" % (epoch, lr_rate, error))
    return weights


def perceptron(train_set, test_set, lr_rate, n_iter):
    weights = train_weights(train_set, lr_rate, n_iter)

    predictions = list()
    for row in test_set:
        predicted = predict(row, weights)
        predictions.append(predicted)
    return predictions


def evaluate_algorithm(dataset: list, algorithm, n_folds, **kwargs):

    folds = k_fold_cross_validation(dataset=dataset, k=n_folds)

    scores = []
    for i in range(len(folds)):
        train_set = list(folds)
        train_set.pop(i)
        train_set = sum(train_set, [])
        test_set = list()
        for row in folds[i]:
            # set actual value to None
            row_copy = row.copy()
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set=train_set, test_set=test_set, **kwargs)
        actual = [row[-1] for row in folds[i]]

        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


if __name__ == "__main__":
    X = impute_missing_values(X)
    X = replace_categorical_values(X)
    X = normalize_data(X)
    y = encode_target_variable(y)

    n_folds = 5
    lr_rate = 0.01
    n_iter = 10
    dataset = pd.concat([X, y], ignore_index=True, axis=1)
    scores = evaluate_algorithm(list(dataset.values), perceptron, n_folds, lr_rate=lr_rate, n_iter=n_iter)
    print("Scores: %s" % scores)
    print("Mean accuracy: %.3f" % (sum(scores) / float(len(scores))))
