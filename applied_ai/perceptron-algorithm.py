from csv import reader
from random import randrange, seed


def predict(row, weights):
    activation = weights[0]

    # skip actual value
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]

    return 1.0 if activation >= 0 else 0.0


# test predictions
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
# for row in dataset:
#     print("Expected=%d, Predicted=%d" % (row[-1], predict(row, weights)))


def train_weights(train, l_rate, n_epoch):

    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            predicted = predict(row, weights)
            expected = row[-1]
            error = (expected - predicted)
            sum_error = error**2
            weights[0] = weights[0] + l_rate * error

            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, l_rate, sum_error))
    return weights


# Calculate weights
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
# l_rate = 0.1
# n_epoch = 5
# weights = train_weights(dataset, l_rate, n_epoch)
# print(weights)


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


dataset = load_csv('../data/raw/connectionist+bench+sonar+mines+vs+rocks/sonar.all-data')


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    for row in dataset:
        row[column] = 1 if row[column] == 'M' else 0


print(dataset[0])
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
print(dataset[0])

str_column_to_int(dataset, len(dataset[0]) - 1)
print(dataset[0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = dataset.copy()

    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if (actual[i] == predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100


# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)

    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)

    return predictions


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        print("Train set before", train_set)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        print("Train set after", train_set)
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            # Set actual value to None in test
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


# Evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 500

scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
