import pandas as pd

file = "/Users/yoon/Downloads/data.csv"
df = pd.read_csv(file)
feature_list = df.columns.tolist()

df.isnull().sum()

# 변수 전처리
selected = ["요일", "시간", "주말여부", "시간대", "유형여부", "B PGM 여부", " MBS_UV\n평균 ",
"PGM", "상품군", "가중분", " 가격 ", " 총주문금액 ", "전환율", " 미리주문 "]

df_selected = df[selected]
df_selected.shape
df_selected.info()

df_selected[' MBS_UV\n평균 '] = df_selected[' MBS_UV\n평균 '].apply( lambda x: x.replace(',', '') )
df_selected[' MBS_UV\n평균 '] = df_selected[' MBS_UV\n평균 '].apply( lambda x: x.replace(' - ', '0') )
df_selected[' 가격 '] = df_selected[' 가격 '].apply( lambda x: x.replace(',', '') )
df_selected[' 가격 '] = df_selected[' 가격 '].apply( lambda x: x.replace(' - ', '0') )
df_selected[' 총주문금액 '] = df_selected[' 총주문금액 '].apply( lambda x: x.replace(',', '') )
df_selected[' 총주문금액 '] = df_selected[' 총주문금액 '].apply( lambda x: x.replace(' - ', '0') )
df_selected['전환율'] = df_selected['전환율'].apply( lambda x: x.replace('%', '') )
df_selected['전환율'] = df_selected['전환율'].apply( lambda x: x.replace(' - ', '0') )
df_selected['전환율'] = df_selected['전환율'].apply( lambda x: x.replace('#DIV/0!', '0') )
df_selected[' 미리주문 '] = df_selected[' 미리주문 '].apply( lambda x: x.replace(',', '') )
df_selected[' 미리주문 '] = df_selected[' 미리주문 '].apply( lambda x: x.replace(' - ', '0') )

reshape_list = [' MBS_UV\n평균 ', ' 가격 ', ' 총주문금액 ', '전환율', ' 미리주문 ']

df_selected.isnull().sum()

df_selected[reshape_list] = df_selected[reshape_list].apply(pd.to_numeric)
df_selected['시간'] = df_selected['시간'].astype(str)

# 변수 시각화
import seaborn as sns

numeric_list = [' MBS_UV\n평균 ', ' 가격 ', ' 총주문금액 ', '전환율', ' 미리주문 ', '가중분']
sns.pairplot(df_selected[numeric_list])

# training
df[' 예상취급액 '] = df[' 예상취급액 '].apply( lambda x: x.replace(',', '') )
df[' 예상취급액 '] = df[' 예상취급액 '].apply(pd.to_numeric)
y = df[' 예상취급액 ']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df_selected_dummies = pd.get_dummies(df_selected)
X_train, X_test, y_train, y_test = train_test_split(df_selected_dummies, y, test_size=0.3, random_state=0)

# regression
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', # mse
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# 학습평가
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# 변수중요소 산출
importances = forest.feature_importances_

feat_labels = X_train.columns.values
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
# 교차검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, X_train, y_train, cv=10)
print(scores)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(Train_X)
X_train_std = sc.transform(Train_X)
X_train_std = pd.DataFrame(X_train_std)
X_train_std.columns = Train_X.columns

X_test_std = sc.transform(Test_X)
X_test_std = pd.DataFrame(X_test_std)
X_test_std.columns = Test_X.columns