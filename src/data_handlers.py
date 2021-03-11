import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np

class DataHandler():

    def split(self, test_size, seed):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(self.dataset, self.labels, test_size=test_size, stratify=self.labels, random_state = seed)


class AdultDataHandler(DataHandler):

    def __init__(self, path):

        dataset = pd.read_csv(path + '/adult.csv', header=None)

        le = sklearn.preprocessing.LabelEncoder()

        self.labels = le.fit_transform(dataset[14])

        dataset = dataset.drop([14], axis=1)
        for i in range(14):
            dataset[i] = dataset.fillna(dataset[i].mode()[0])
        dataset = dataset.to_numpy()
        dataset[:, 1] = le.fit_transform(dataset[:, 1])
        dataset[:, 3] = le.fit_transform(dataset[:, 3])
        dataset[:, 5] = le.fit_transform(dataset[:, 5])
        dataset[:, 6] = le.fit_transform(dataset[:, 6])
        dataset[:, 7] = le.fit_transform(dataset[:, 7])
        dataset[:, 8] = le.fit_transform(dataset[:, 8])
        dataset[:, 9] = le.fit_transform(dataset[:, 9])
        dataset[:, 13] = le.fit_transform(dataset[:, 13])

        self.dataset = dataset

        self.feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital_status',
                              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                              'hours-per-week', 'native-country']

        self.class_names = ['<=50K', '>50K']

class BanknoteDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/data_banknote_authentication.csv')
        self.labels = dataset["Col4"].to_numpy()
        dataset = dataset.drop(["Col4"], axis=1)
        self.feature_names = dataset.columns
        self.dataset = dataset.to_numpy()
        self.class_names = ['0', '1']

class DiabetesDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/diabetes.csv')
        self.labels = dataset["Outcome"].to_numpy()
        dataset = dataset.drop(["Outcome"], axis=1)
        self.feature_names = dataset.columns
        self.dataset = dataset.to_numpy()
        self.class_names = ['Healthy','Diabetes']

class DigitsDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/digits_train.csv')
        dataset = dataset.append(pd.read_csv(path + '/digits_test.csv'))
        self.labels = dataset["Col16"].to_numpy()
        dataset = dataset.drop(["Col16"], axis=1)
        self.feature_names = dataset.columns
        self.dataset = dataset.to_numpy()
        self.class_names = ['0', '1', '2', '3', '4', '5', '6' ,'7', '8', '9']

class MagicDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/magic04.csv', header=None)
        dataset[10][dataset[10] == 'g'] = 1
        dataset[10][dataset[10] == 'h'] = 0
        dataset[10] = dataset[10].astype(int)
        labels = dataset[10].to_numpy()
        dataset = dataset.drop([10], axis=1)
        self.feature_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
                              'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
        self.dataset = dataset.to_numpy()
        labels[labels == 'g'] = 1
        labels[labels == 'h'] = 0
        self.labels = labels.astype(int)
        self.class_names = ['hadron', 'gamma']

class SpambaseDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/spambase.csv', header=None)
        self.labels = dataset[57].to_numpy()
        dataset = dataset.drop([57], axis=1)
        self.feature_names = dataset.columns
        self.dataset = dataset.to_numpy()
        f = open(path + '/spambase.names', 'r')
        lines = f.readlines()[-57:]
        self.feature_names = [lines[i].split(':')[0] for i in range(len(lines))]
        f.close()
        self.class_names = ['not spam', 'spam']

class TitanicDataHandler(DataHandler):

    def __init__(self, path):
        dataset = pd.read_csv(path + '/titanic.csv')[:890]
        self.labels = dataset["Survived"].to_numpy().astype(int)

        dataset = dataset.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

        values = {'Age': dataset['Age'].mean(axis=0), 'Fare': dataset['Fare'].mean(axis=0), 'Embarked': 'C'}
        dataset = dataset.fillna(value=values)

        self.feature_names = dataset.columns
        dataset = dataset.to_numpy()

        # categorical features encoding
        le = sklearn.preprocessing.LabelEncoder()
        dataset[:, 1] = le.fit_transform(dataset[:, 1])
        dataset[:, -1] = le.fit_transform(dataset[:, -1])
        self.dataset = dataset
        self.class_names = ['not survived', 'survived']