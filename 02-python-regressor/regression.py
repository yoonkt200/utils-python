import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

income_list = [119,85,97,95,120,92,105,110,98,98,81,81,81,91,105,100,
               107,82,84,100,108,116,115,93,105,89,104,108,88,109,112,
               96,89,93,114,81,84,88,96,82]
consump_list = [154,123,125,130,151,131,141,141,130,134,115,117,123,144,137,
                140,123,115,134,147,144,144,126,141,124,144,144,129,137,144,
                132,125,126,140,120,118,119,131,127,150]

# 훈련데이터로 학습시키기
regr = linear_model.LinearRegression()

income_list_ = np.asarray(income_list)
consump_list_ = np.asarray(consump_list)
income_list_fit = income_list_.reshape(-1,1)
consump_list_fit = consump_list_.reshape(-1,1)

regr.fit(income_list_fit, consump_list_fit)

# 테스트 데이터로 결과값 예측
y_pred = regr.predict(income_list_fit)

# 회귀 계수 출력
print('Coefficients: \n', regr.coef_)

# MSE 출력
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))

# r-score 출력
print('Variance score: %.2f' % r2_score(y_test, y_pred))
plt.scatter(income_list, consump_list,  color='black')
plt.plot(income_list, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
