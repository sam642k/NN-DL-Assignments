import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, KFold

x= np.random.normal(0, 1, 5000)
eps= np.random.normal(0, 0.25, 5000)
y= -1 + 0.5*x -2*x**2 +0.3*x**3 + eps
# plt.scatter(x,y)
# plt.show()
x=x.reshape(-1 ,1)
kf= KFold(n_splits=10)
yhats=[]
y_tests=[]
for traini, testi in kf.split(x):
    x_train, x_test, y_train, y_test= x[traini], x[testi], y[traini], y[testi]
    model= LinearRegression()

    model.fit(x_train, y_train)
    yhat= model.predict(x_test)
    yhats.append(yhat)
    y_tests.append(y_test)

print("Average Score is ", r2_score(y_tests, yhats))