import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def preprocess_telco():
    dataset = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # # check for unique categorical values in each column
    # for col in dataset.columns:
    #     if dataset[col].dtype == 'object':
    #         print(f'{col}: {dataset[col].unique()} {dataset[col].unique().size}')


    dataset.drop('customerID', axis=1, inplace=True)

    X = dataset.drop('Churn', axis=1)
    y = dataset['Churn']

    y = y.map({'Yes': 1, 'No': 0})

    # Total Charges should be converted from string to float
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X['TotalCharges'].dtype

    # print(dataset.isnull().sum())

    # For Telco, missing values in TotalCharges column means 0
    X['TotalCharges'].fillna(0.0, inplace=True)

    X['MultipleLines'].replace('No phone service', 'No', inplace=True)

    X = X.copy()
    # Iterate over each column in the dataset
    for col in X.columns.values:
        if X[col].dtype == 'object':
            # Take care of the Yes/No columns
            if X[col].unique().size == 2 and 'Yes' in X[col].unique() and 'No' in X[col].unique():
                # How to make sure Yes is encoded as 1 and No is encoded as 0?
                X[col] = X[col].map({'Yes': 1, 'No': 0})
                
    # One hot encode the the rest of the categorical columns
    X = pd.get_dummies(X, dtype=np.int64)

    # Split the last column for label array and rest for training array using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)

    # print(f'X_train: {X_train.shape}')
    # print(f'X_test: {X_test.shape}')

    # Normalize the training array
    scaler = MinMaxScaler()
    for col in X_train.columns.values:
        X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
        X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
        
    return X_train, X_test, y_train, y_test

def preprocess_adult():
    train_dataset = pd.read_csv('dataset/adult/adult.data', header=None)
    test_dataset = pd.read_csv('dataset/adult/adult.test', header=None, skiprows=1)

    # add a flag to identify train and test data
    train_dataset['train'] = 1
    test_dataset['train'] = 0

    # concatenate train and test data
    dataset = pd.concat([train_dataset, test_dataset], axis=0)
    print(dataset.shape)

    # use these column names
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race',
                    'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income', 'train']

    dataset.columns = column_names

    # check for unique categorical values in each column
    # for col in dataset.columns:
    #     if dataset[col].dtype == 'object':
    #         print(f'{col}: {dataset[col].unique()} {dataset[col].unique().size}')

    # check for missing values
    # handle ? in workclass, occupation and native-country with categorical imputer
    categorical_imputer = SimpleImputer(strategy='most_frequent', missing_values=' ?')
    columns = ['workclass', 'occupation', 'native-country']
    dataset[columns] = categorical_imputer.fit_transform(dataset[columns])

    # # check for unique categorical values in each column
    # cont_columns = []
    # for col in dataset.columns:
    #     if dataset[col].dtype == 'object':
    #         print(f'{col}: {dataset[col].unique()} {dataset[col].unique().size}')
    #     else:
    #         cont_columns.append(col)

    # change income column to binary and name to 'income-over-50k'
    dataset['income'] = dataset['income'].apply(lambda x: 1 if x == ' >50K' or x == ' >50K.' else 0)
    dataset.rename(columns={'income': 'income-over-50k'}, inplace=True)

    # one hot encode categorical columns
    dataset = pd.get_dummies(dataset, dtype=np.int64, drop_first=True)
            
    # separate train and test data
    train_dataset = dataset[dataset['train'] == 1]
    test_dataset = dataset[dataset['train'] == 0]

    # drop train and test flag columns
    train_dataset = train_dataset.drop(columns=['train'])
    test_dataset = test_dataset.drop(columns=['train'])

    # separate features and labels
    X_train = train_dataset.drop(columns=['income-over-50k'])
    y_train = train_dataset['income-over-50k']

    X_test = test_dataset.drop(columns=['income-over-50k'])
    y_test = test_dataset['income-over-50k']

    # print(cont_columns)

    # Normalize
    scaler = MinMaxScaler()
    for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
        X_test[col] = scaler.fit_transform(X_test[col].values.reshape(-1, 1))
        
    return X_train, X_test, y_train, y_test

def preprocess_creditcard():
    dataset = pd.read_csv('dataset/creditcard.csv')

    # check for unique categorical values in each column
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            print(f'{col}: {dataset[col].unique()} {dataset[col].unique().size}')

    # # check for missing values
    # dataset.isnull().sum()

    # check for class imbalance
    print('before sampling', dataset['Class'].value_counts())

    # separate ones and zeros
    ones = dataset[dataset['Class'] == 1]
    zeros = dataset[dataset['Class'] == 0]

    # # plot class imbalance
    # plt.bar(['0', '1'], [zeros.shape[0], ones.shape[0]])
    # plt.xlabel('Class')
    # plt.ylabel('Number of transactions')
    # plt.title('Class imbalance')
    # plt.show()

    # take smaller subset of data so that the data is balanced
    zeros = zeros.sample(n=20000, random_state=112)

    # combine the two subsets
    dataset = pd.concat([zeros, ones], axis=0)
    print('after sampling', dataset['Class'].value_counts())

    # shuffle the dataset
    dataset = dataset.sample(frac=1, random_state=112).reset_index(drop=True)

    # separate features and labels
    X = dataset.drop(columns=['Class'])
    y = dataset['Class']

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)
    print(f'train set 0/1', y_train.value_counts())
    print(f'test set 0/1', y_test.value_counts())

    return X_train, X_test, y_train, y_test


def entropy(y):
    if len(y) == 0:
        return 0
    
    p = np.sum(y) / len(y)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def remainder(feature, y):
    # Calculate the remainder
    remainder = 0
    for val in np.unique(feature):
        y_subset = y[feature == val]
        remainder += len(y_subset) / len(y) * entropy(y_subset)
    return remainder
    
def information_gain(feature, y):
    # for binary features
    if np.unique(feature).size == 2:
        return entropy(y) - remainder(feature, y)
    # for continuous features
    else:
        sorted_indices = np.argsort(feature)
        feature = feature[sorted_indices]
        y = y[sorted_indices]
        split_points = (feature[1:] + feature[:-1]) / 2
        
        # find the best split
        max_info_gain = 0
        split_point = 0
        for val in split_points:
            y_left = y[feature <= val]
            y_right = y[feature > val]
            info_gain = entropy(y) - ( len(y_left) / len(y) * entropy(y_left) - len(y_right) / len(y) * entropy(y_right) )
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_point = val
                
        return max_info_gain

def feature_selection(X, y, num_features):
    if num_features > X.shape[1]:
        # raise ValueError('num_features cannot be greater than number of features in X')
        num_features = X.shape[1]
    
    # Calculate information gain for each feature
    info_gain = []
    for col in X.columns:
        info_gain.append(information_gain(X[col].to_numpy(), y.to_numpy()))
        
    # Sort the features in descending order of information gain
    sorted_features = np.argsort(info_gain)[::-1]
    top_features = sorted_features[:num_features]
    
    # Return the subset with the top num_features and their names
    return X.iloc[:, top_features], X.columns[top_features]

def sigmoid(z):
    # to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0/(1.0+np.exp(-z))
    
def cost_function(X, y, w):
    # Calculate the cost
    z = np.dot(X, w)
    y_pred = sigmoid(z)
    # to avoid divide by zero
    y_pred = np.clip(y_pred, 1e-16, 1 - 1e-16)
    cost = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

def gradient_descent(X, y, alpha, num_iter, threshold):
    # Initialize weights
    w = np.zeros(X.shape[1])
    w0 = 0
    
    # Iterate for num_iter times
    for i in range(num_iter):
        # Calculate the gradient
        y_pred = sigmoid(np.dot(X, w) + w0)
        gradient = np.dot(X.T, (y - y_pred))
        gradient0 = np.sum(y - y_pred)
        
        # Update the weights
        w += alpha * gradient
        w0 += alpha * gradient0
        
        # Calculate the error
        error = cost_function(X, y, w)
        
        # Early terminate if error < threshold
        if error < threshold:
            break
        
    return w, w0

class LogisticRegressionWeak:

    def fit(self, X_train, y_train, alpha, num_iter, threshold):
        # Train the model
        w, w0 = gradient_descent(X_train, y_train, alpha, num_iter, threshold)
        self.w = w
        self.w0 = w0
        
    def predict(self, X):
        # Predict the labels
        y_pred = sigmoid(np.dot(X, self.w) + self.w0)
        return y_pred
    
class Adaboost:
    def boost(self, X_train, y_train, K):
        # returns a weighted majority hypothesis
        # X_train, y_train, X_test, y_test: training and test data
        # logistic_regression: a learning algorithm
        # K: the number of hypotheses in the ensemble
        
        N = X_train.shape[0] // 2
        
        # w, a vector of N example weights, initially 1/N
        w = np.ones(N) / N
        # h, a vector of K hypotheses
        h = []
        # z, a vector of K hypothesis weights
        z = []
        
        examples = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        for _ in range(K):
            # d ← Resample(X_train, w)
            d = examples[np.random.choice(N, N, p=w, replace=True)]
            # print(d)
            d_x = d[:, :-1]
            d_y = d[:, -1]
            # print(d_x, d_y, d_x.shape, d_y.shape)
            # h[k] ← logistic_regression(d)
            # model = LogisticRegressionWeak()
            model = LogisticRegression()
            h.append(model)
            # model.fit(d_x, d_y, alpha=0.1, num_iter=1000, threshold=0.1)
            model.fit(d_x, d_y)
            y_pred = model.predict(d_x)
            
            # # error ← 0
            # error = 0
            # for j in range(len(d_y)):
            #     # if h[k](xj) ≠ yj
            #     if y_pred[j] != d_y[j]:
            #         # error ← error + w[j]
            #         error += w[j]
            error = np.sum(w[y_pred != d_y])
                    
            # if error > .5 then continue
            if error > 0.5:
                z.append(0)
                # h.pop()
                print('error is', error)
                continue

            # for j ← 1 to N do
            # if h[k](xj) = yj then w[j] ← w[j] · error/(1 − error)
            for j in range(len(d_y)):
                if y_pred[j] != d_y[j]:
                    w[j] *= error / (1 - error)
                    
            # w ← Normalize(w)
            w /= np.sum(w)
            print(w)
            
            # z[k] ← log((1 − error)/error)
            z.append(np.log((1 - error) / error))
            
        self.hypotheses = np.array(h)
        self.hyp_weights = np.array(z)
        # print(self.hypotheses, self.hyp_weights)
            
        # return Weighted_Majority(h, z)
        return
    
    def weighted_majority_predict(self, X):
        # returns a weighted majority hypothesis
        
        # h ← the zero hypothesis
        h = np.zeros(X.shape[0])
        # print(h.shape)
        # for k ← 1 to K do
        for k in range(len(self.hypotheses)):
            # h ← h + z[k] · h[k]
            h += self.hyp_weights[k] * self.hypotheses[k].predict(X)
        
        # print(h)
        # return h
        return h
    
def report_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, np.round(y_pred))
    print(f'Accuracy: {accuracy}')
    
def report_recall(y_test, y_pred):
    # recall or sensitivity or true positive rate or hit rate = TP / (TP + FN)
    if np.sum(y_test == 1) == 0:
        recall = 0
    else:
        recall = np.sum(np.logical_and(y_test == 1, np.round(y_pred) == 1)) / np.sum(y_test == 1)
    print(f'Recall: {recall}')
    
def report_specificity(y_test, y_pred):
    # true negative rate or specificity = TN / (TN + FP)
    if np.sum(y_test == 0) == 0:
        specificity = 0
    else:
        specificity = np.sum(np.logical_and(y_test == 0, np.round(y_pred) == 0)) / np.sum(y_test == 0)
    print(f'Specificity: {specificity}')
    
def report_precision(y_test, y_pred):
    # true positive rate or precision = TP / (TP + FP)
    if np.sum(np.round(y_pred) == 1) == 0:
        precision = 0
    else:
        precision = np.sum(np.logical_and(y_test == 1, np.round(y_pred) == 1)) / np.sum(np.round(y_pred) == 1)
    print(f'Precision: {precision}')
    
def report_false_discovery_rate(y_test, y_pred):
    # false discovery rate = FP / (FP + TP)
    if np.sum(np.round(y_pred) == 1) == 0:
        fdr = 0
    else:
        fdr = np.sum(np.logical_and(y_test == 0, np.round(y_pred) == 1)) / np.sum(np.round(y_pred) == 1)
    print(f'False discovery rate: {fdr}')
    
def report_f1_score(y_test, y_pred):
    # f1 score = 2 * precision * recall / (precision + recall)
    if np.sum(np.round(y_pred) == 1) == 0 or np.sum(y_test == 1) == 0:
        f1_score = 0
    else:
        precision = np.sum(np.logical_and(y_test == 1, np.round(y_pred) == 1)) / np.sum(np.round(y_pred) == 1)
        recall = np.sum(np.logical_and(y_test == 1, np.round(y_pred) == 1)) / np.sum(y_test == 1)
        f1_score = 2 * precision * recall / (precision + recall)
    print(f'F1 score: {f1_score}')
    
def report_all(y_test, y_pred):
    report_accuracy(y_test, y_pred)
    report_recall(y_test, y_pred)
    report_specificity(y_test, y_pred)
    report_precision(y_test, y_pred)
    report_false_discovery_rate(y_test, y_pred)
    report_f1_score(y_test, y_pred)

# X_train, X_test, y_train, y_test = preprocess_telco()
# X_train, X_test, y_train, y_test = preprocess_adult()
X_train, X_test, y_train, y_test = preprocess_creditcard()

# hyperparameters
alpha = 0.01
num_iter = 5000
threshold = 0.1
num_features = 20
num_hyp = 7

# # Feature selection
X_train_subset, top_features = feature_selection(X_train, y_train, num_features)
X_test_subset = X_test[top_features]

# model = LogisticRegressionWeak()
# model.fit(X_train_subset.to_numpy(), y_train.to_numpy(), alpha, num_iter, threshold)
# print("Performance of Weak Logistic Regression on Training Data")
# y_pred = model.predict(X_train_subset.to_numpy())
# report_all(y_train, y_pred)
# print("Performance of Weak Logistic Regression on Test Data")
# y_pred = model.predict(X_test_subset.to_numpy())
# report_all(y_test, y_pred)

np.random.seed(112)
model = Adaboost()
model.boost(X_train_subset.to_numpy(), y_train.to_numpy(), num_hyp)
print("Performance of Adaboost on Training Data")
y_pred = model.weighted_majority_predict(X_train_subset.to_numpy())
report_all(y_train, y_pred)
print("Performance of Adaboost on Test Data")
y_pred = model.weighted_majority_predict(X_test_subset.to_numpy())
report_all(y_test, y_pred)