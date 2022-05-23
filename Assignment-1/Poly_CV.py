import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

for d in range(1, 10):
    np.random.seed(56)
    print("for d= ", d)
    x = np.random.normal(0, 1, 5000)
    print(x[0])
    eps = np.random.normal(0, 0.25, 5000)
    y = -1 + 0.5 * x - 2 * x ** 2 + 0.3 * x ** 3 + eps
    x = x.reshape(-1, 1)

    poly = PolynomialFeatures(d)
    x = poly.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    kf= KFold(n_splits=10)
    model = LinearRegression()
    for traini, validi in kf.split(x_train):
        x_train, x_valid, y_train, y_valid= x_train[traini], x_train[validi], y_train[traini], y_train[validi]
        model = LinearRegression()

        model.fit(x_train, y_train)
        ycv= model.predict(x_valid)

        yhat = model.predict(x_test)
        print("Test set Accuracy: ", r2_score(y_test, yhat))