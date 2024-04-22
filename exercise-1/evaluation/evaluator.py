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



# data_sets = {
#     'dataset_1': {
#         'alpha:0.001': {'accuracy': 0.965034965034965, 'recall': 0.965034965034965, 'precision': 0.9652663191989034,
#                         'f1': 0.9651006853533873, 'time': 280.17568588256836, 'confusion_matrix': [[87, 3], [2, 51]],
#                         'alpha': 0.001, 'hidden_layer_sizes': (75, 75, 75)},
#         'alpha:0.0001': {'accuracy': 0.972027972027972, 'recall': 0.972027972027972, 'precision': 0.972027972027972,
#                          'f1': 0.972027972027972, 'time': 339.1234874725342, 'confusion_matrix': [[88, 2], [2, 51]],
#                          'alpha': 0.0001, 'hidden_layer_sizes': (100, 100, 100)},
#         'alpha:1e-05': {'accuracy': 0.972027972027972, 'recall': 0.972027972027972, 'precision': 0.972027972027972,
#                         'f1': 0.972027972027972, 'time': 341.6876792907715, 'confusion_matrix': [[88, 2], [2, 51]],
#                         'alpha': 1e-05, 'hidden_layer_sizes': (120, 120, 120)}
#     },
#     'dataset_2': {
#         'alpha:0.001': {'accuracy': 0.965034965034965, 'recall': 0.865034965034965, 'precision': 0.9652663191989034,
#                         'f1': 0.9651006853533873, 'time': 280.17568588256836, 'confusion_matrix': [[87, 3], [2, 51]],
#                         'alpha': 0.001, 'hidden_layer_sizes': (75, 75, 75)},
#         'alpha:0.0001': {'accuracy': 0.972027972027972, 'recall': 0.872027972027972, 'precision': 0.972027972027972,
#                          'f1': 0.972027972027972, 'time': 339.1234874725342, 'confusion_matrix': [[88, 2], [2, 51]],
#                          'alpha': 0.0001, 'hidden_layer_sizes': (100, 100, 100)},
#         'alpha:1e-05': {'accuracy': 0.982027972027972, 'recall': 0.872027972027972, 'precision': 0.972027972027972,
#                         'f1': 0.972027972027972, 'time': 341.6876792907715, 'confusion_matrix': [[88, 2], [2, 51]],
#                         'alpha': 1e-05, 'hidden_layer_sizes': (130, 130, 130)}
#     },
# }

def draw_diagram2(evaluation_results, x_axis="alpha", y_axis="accuracy", figsize=(10, 6), title=None, logaritmic=False, line=True):
    plt.figure(figsize=figsize)

    is_str = False
    for dataset_name, results in evaluation_results.items():

        x_values = [metrics[x_axis] for metrics in results.values()]
        y_values = [metrics[y_axis] for metrics in results.values()]
        if isinstance(x_values[0], (int, float, complex)):
            plt.plot(x_values, y_values, marker='o', label=dataset_name)
        else:
            is_str = True
            # Extract key-value pairs and sort based on the x_metric lexicographically
            items = [(config[x_axis], config[y_axis]) for config in results.values()]
            items.sort()  # This sorts tuples lexicographically by default

            # Split sorted items back into x and y values
            x_values, y_values = zip(*items)

            # Convert tuples to string labels if necessary
            x_labels = [str(x) if isinstance(x, tuple) else x for x in x_values]

            if line:
                plt.plot(x_labels, y_values, marker='o', label=dataset_name)
            else:
                plt.scatter(x_labels, y_values, marker='o', label=dataset_name)


    if title is not None:
        plt.title(title)
    else:
        plt.title(y_axis.capitalize())
    # make x-axis logarithmic
    if logaritmic:
        plt.xscale('log')

    plt.xlabel(x_axis.capitalize())
    plt.ylabel(y_axis.capitalize())
    if is_str:
        plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_diagram2_list(evaluation_results, x_axis="alpha", y_axis=["accuracy", "precision"], figsize=(10, 6), title=None, logaritmic=False, line=True):

    for i, y in enumerate(y_axis):
        draw_diagram2(evaluation_results, x_axis, y, figsize, title[i] if title is not None else None, logaritmic, line)


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
