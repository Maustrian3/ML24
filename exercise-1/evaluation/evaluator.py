import time

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate
import pandas as pd

import warnings


def evaluate(
        classifier,
        X_train,
        y_train,
        X_test,
        y_test,
        hyperparameters={},
        hyperparameter_iterate="",
        hyperparameter_iterations=[]
):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print("================================================================================")
    print("Evaluating classifier: ", classifier.__name__)
    print("Hyperparameters: ", hyperparameters)

    # no iteration values are selected only execute one run
    if len(hyperparameter_iterations) == 0:
        clf = classifier(**hyperparameters)
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        y_pred = clf.predict(X_test)
        __eval(clf, y_test, y_pred, start, end, "", "")

    for hyperparameter_value in hyperparameter_iterations:
        hp = hyperparameters.copy()
        hp[hyperparameter_iterate] = hyperparameter_value
        clf = classifier(**hp)
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        y_pred = clf.predict(X_test)
        __eval(clf, y_test, y_pred, start, end, hyperparameter_iterate, hyperparameter_value)

    print("================================================================================")

def __eval(clf, y_test, y_pred, start, end, hyperparameter_iterate, hyperparameter_value):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    print("========================================")
    if hyperparameter_iterate != "":
        print("Hyperparameter ", hyperparameter_iterate, " value: ", hyperparameter_value)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
    print("Time: {:.6f}ms".format((end - start) * 1000))
    print("Confusion Matrix: ")
    print(tabulate(cm, headers='keys', tablefmt='psql'))
