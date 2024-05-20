import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# def gradient_descent(X, y, alpha, epochs):
# Initialize parameters to random? value

# In each iteration
# Make predictions with current model
# Calculate the new gradient
# The derivatives in respect to the known features
# Update the parameters with the new gradient

# TODO stepX: Also programmatically derive loss functions? Is this needed? Or just use 2-3 different ones and calc them beforehand.
class MySGDRegressor:
    # Loss Function: Sum of Squared Residuals
    # sum_squared_residuals = (observed - predicted)^2 = (observed - (intercept + slope * cur_predicted))^2

    def __init__(self, max_iter=1000, alpha=0.0001, intercept=0):
        self.max_iter = max_iter
        self.alpha = alpha
        self.intercept = intercept
        self.coefficients = None

    def __init_params(self, n_features):
        self.intercept = 0
        self.coefficients = np.zeros(n_features)

    def __update_gradients(self, X, y):
        # Calculate new slope and intercept
        # Loss Function: Sum of Squared Residuals
        # Calculate derivative in respect to the slope
        # derivative(sum_squared_residuals) = -2 * (observed - (intercept + slope * cur_predicted))

        n_samples, n_features = X.shape
        intercept_gradient = 0
        coefficients_gradient = 0

        predictions = self.intercept + X.dot(self.coefficients)
        errors = y - predictions

        intercept_gradient += -2 * np.sum(errors) / n_samples
        coefficients_gradient += -2 * X.T.dot(errors) / n_samples

        step_size_intercept = intercept_gradient * self.alpha
        step_size_slope = coefficients_gradient * self.alpha

        # Calc new slope/intercept values
        self.intercept = self.intercept - step_size_intercept
        self.coefficients = self.coefficients - step_size_slope

    def fit(self, X, y):
        self.__init_params(X.shape[1])
        for _ in range(self.max_iter):
            self.__update_gradients(X, y)

    def predict(self, X):
        return self.intercept + np.dot(X, self.coefficients)


def main():
    # data = arff.loadarff('./datasets/black_friday.arff')
    # df = pd.DataFrame(data[0])

    california = fetch_california_housing()
    X = california.data
    y = california.target

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # splitting dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    max_iter = 1000
    alpha = 0.0001

    # Least squares regression model
    ls_model = LinearRegression()
    ls_model.fit(X_train, y_train)
    ls_y_pred = ls_model.predict(X_test)

    sgd_regressor = SGDRegressor(max_iter=max_iter, alpha=alpha, eta0=alpha,
                                 loss='squared_error',
                                 learning_rate='constant', tol=None,
                                 shuffle=False,
                                 penalty=None
                                 )
    sgd_regressor.fit(X_train, y_train)
    sgd_y_pred = sgd_regressor.predict(X_test)

    my_sgd_regressor = MySGDRegressor(max_iter=max_iter, alpha=alpha)
    my_sgd_regressor.fit(X_train, y_train)
    my_y_pred = my_sgd_regressor.predict(X_test)

    # Calc mean squared error for evaluation
    ls_mse = mean_squared_error(y_test, ls_y_pred)
    sgd_mse = mean_squared_error(y_test, sgd_y_pred)
    my_mse = mean_squared_error(y_test, my_y_pred)
    print("sklearn ls regressor: ", ls_mse)
    print("sklearn sgd regressor: ", sgd_mse)
    print("my regressor: ", my_mse)


if __name__ == '__main__':
    main()
