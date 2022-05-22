import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

x= np.random.normal(0, 1, 5000)
X=[]
for a in x:
    X.append([a**0, a, a**2, a**3])
X=np.array(X)
eps= np.random.normal(0, 0.25, 5000)
W= [-1, 0.5, -2, 0.3]

y=[]
for i in range(5000):
    k=0
    for j in range(4):
        k+=W[j]*X[i][j]
    y.append(k+eps[i])
y=np.array(y)

loo= LeaveOneOut()

errors=[]
weights=[]
yhats=[]
y_tests=[]
for traini, testi in loo.split(X):
    x_train, x_test, y_train, y_test= X[traini], X[testi], y[traini], y[testi]
    model= LinearRegression()
    model.fit(x_train, y_train)
    yhat= model.predict(x_test)
    errors.append(mean_squared_error(y_test,yhat))
    weights.append(model.coef_)
    yhats.append(yhat)
    y_tests.append(y_test)

print("Average error is ", sum(errors)/len(errors))
print("True Weights: ", W)
print("Average weights: ",sum(weights)/len(weights))
print("Average score is ", r2_score(y_tests, yhats))