import time

from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # load diabetes dataset
    from sklearn.datasets import load_diabetes, make_regression
    import numpy as np

    #diabetes = load_diabetes()
    #X = diabetes.data
    #y = diabetes.target

    X, y = make_regression(n_samples=2000, n_features=25, noise=1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from knn import CustomKNeighborsRegressor
    brute_knn = CustomKNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='minkowski')
    # measure the time taken to fit the model
    start = time.time()
    brute_knn.fit(X_train, y_train)
    y_pred = brute_knn.predict(X_test)
    end = time.time()
    print("Time taken to fit the model brute: ", end - start)

    kd_tree_knn = CustomKNeighborsRegressor(n_neighbors=5, algorithm='kd_tree', metric='minkowski', leaf_size=40)
    start = time.time()
    kd_tree_knn.fit(X_train, y_train)
    y_pred_kd = kd_tree_knn.predict(X_test)
    end = time.time()
    print("Time taken to fit the model tree: ", end - start)

    # compare the predictions
    print(np.array_equal(y_pred, y_pred_kd))

    # use the sklearn KNeighborsRegressor
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='minkowski')
    start = time.time()
    knn.fit(X_train, y_train)
    y_pred_sklearn = knn.predict(X_test)
    end = time.time()
    print("Time taken to fit the model sklearn brute: ", end - start)

    print(np.array_equal(y_pred, y_pred_sklearn))


