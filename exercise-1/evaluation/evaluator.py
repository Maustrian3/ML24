import pandas as pd
import time
import warnings
from matplotlib import pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tabulate import tabulate


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


def evaluate2(
        classifier,
        X_train,
        y_train,
        X_test,
        y_test,
        hyperparameters={},
        hyperparameters_iterate={},
):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print("================================================================================")
    print("Evaluating classifier: ", classifier.__name__)
    print("Hyperparameters: ", hyperparameters)

    # no iteration values are selected only execute one run
    if len(hyperparameters_iterate) == 0:
        clf = classifier(**hyperparameters)
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        y_pred = clf.predict(X_test)
        return {'default': __eval(clf, y_test, y_pred, start, end, "", "")}

    results = {}
    for hyperparameter, iteration_values in hyperparameters_iterate.items():
        for hyperparameter_value in iteration_values:
            hp = hyperparameters.copy()
            hp[hyperparameter] = hyperparameter_value
            clf = classifier(**hp)
            start = time.time()
            clf.fit(X_train, y_train)
            end = time.time()
            y_pred = clf.predict(X_test)
            results[str(hyperparameter) + ":" + str(hyperparameter_value)] = __eval(clf, y_test, y_pred, start, end,
                                                                                    hyperparameter,
                                                                                    hyperparameter_value)

    print("================================================================================")
    return results


def __eval(clf, y_test, y_pred, start, end, hyperparameter_iterate, hyperparameter_value):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    evaluation_results = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "time": (end - start) * 1000,
        "confusion_matrix": cm
    }

    print("========================================")
    if hyperparameter_iterate != "":
        print("Hyperparameter ", hyperparameter_iterate, " value: ", hyperparameter_value)
        evaluation_results[hyperparameter_iterate] = hyperparameter_value
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
    print("Time: {:.6f}ms".format((end - start) * 1000))
    print("Confusion Matrix: ")
    print(tabulate(cm, headers='keys', tablefmt='psql'))
    return evaluation_results


# Define a generator function to yield all fields from the dataset
def __extract_fields(data_sets):
    for dataset in data_sets.values():
        for entry in dataset.values():
            yield from entry.keys()


def draw_diagrams(evaluation_results):
    # Use a set comprehension to collect unique fields
    fields_set = {
        field
        for field in __extract_fields(evaluation_results)
        if field != 'alpha' and field != 'confusion_matrix'
    }
    for field in fields_set:
        __draw_diagram_for_field(field, evaluation_results)


def __draw_diagram_for_field(field, evaluation_results, figsize=(10, 6)):
    # Plotting
    plt.figure(figsize=figsize)

    plt.figure(figsize=figsize)

    for dataset_name, dataset in evaluation_results.items():
        hyperparameter_values = []
        values = []

        for key, value in dataset.items():
            hyperparameter_values.append(value[key.split(":")[0]])
            values.append(value[field])

        plt.plot(hyperparameter_values, values, marker='o', label=dataset_name)

    plt.title(field.capitalize())
    plt.xlabel('Alpha')
    plt.ylabel(field.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()
