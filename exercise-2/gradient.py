import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# def gradient_descent(X, y, alpha, epochs):
# Initialize parameters to random? value

# In each iteration
# Make predictions with current model
# Calculate the new gradient
# The derivatives in respect to the known features
# Update the parameters with the new gradient

# TODO step1: implement basic with 1 feature -> easier to understand
# TODO stepX: Also programmatically derive loss functions? Is this needed? Or just use 2-3 different ones and calc them beforehand.
class MySGDRegressor:
    # Loss Function: Sum of Squared Residuals
    # sum_squared_residuals = (observed - predicted)^2 = (observed - (intercept + slope * cur_predicted))^2

    def __init__(self, max_iter=1000, alpha=0.0001, intercept=0, slope=1):
        self.slope = slope
        self.intercept = intercept
        self.max_iter = max_iter
        self.alpha = alpha

    def init_params(self, intercept=0, slope=1):
        self.intercept = intercept
        self.slope = slope

    def __update_gradients(self, X, y):
        # Calculate new slope and intercept
        # Loss Function: Sum of Squared Residuals
        # Calculate derivative in respect to the slope
        # derivative(sum_squared_residuals) = -2 * (observed - (intercept + slope * cur_predicted))
        intercept_gradient = 0
        slope_gradient = 0
        for i in range(len(X)):  # For each sample do:
            # calc the value of the current sample
            # TODO change to handle more than 1 feature
            intercept_gradient += -2 * (y[i] - (self.intercept + self.slope * X[i]))
            slope_gradient += -2 * X[i] * (y[i] - (self.intercept + self.slope * X[i]))
            # sum up
        # Resulting in a 'slope' for the step size calculation of the slope
        # Plug in into step size formula for the slope
        step_size_intercept = (intercept_gradient / len(X)) * self.alpha
        step_size_slope = (slope_gradient / len(X)) * self.alpha

        # Calc new slope/intercept values
        self.intercept = self.intercept - step_size_intercept
        self.slope = self.slope - step_size_slope

    def fit(self, X, y):
        for _ in range(self.max_iter):
            self.__update_gradients(X, y)

    def predict(self, X):
        return self.intercept + self.slope * np.array(X)


def main():
    # TODO use specified dataset
    # data = arff.loadarff('./datasets/black_friday.arff')
    # df = pd.DataFrame(data[0])

    california = fetch_california_housing()
    X = california.data
    y = california.target

    # TODO make sure X has only 1 feature
    X = X[:, 0]
    X = X.reshape((-1, 1))

    # splitting dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_iter = 200
    alpha = 0.001

    sgd_regressor = SGDRegressor(max_iter=max_iter, alpha=alpha, learning_rate='invscaling', random_state=42)
    sgd_regressor.fit(X_train, y_train)
    y_pred = sgd_regressor.predict(X_test)

    my_sgd_regressor = MySGDRegressor(max_iter=max_iter, alpha=alpha)
    my_sgd_regressor.fit(X_train, y_train)
    my_y_pred = my_sgd_regressor.predict(X_test)

    # Calc mean squared error for evaluation
    mse = mean_squared_error(y_test, y_pred)
    my_mse = mean_squared_error(y_test, my_y_pred)
    print(mse)
    print(my_mse)


if __name__ == '__main__':
    main()
