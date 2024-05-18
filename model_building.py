import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/eda_data.csv')

# Choose relevant columns
df_model = df[['avg_salary','Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'comp_count', 'hourly', 'employer', 'job_state', 'same_state', 'age_of_company', 'Python', 'ML', 'AWS', 'Excel', 'DL', 'job_title', 'seniority', 'desc_len']]
# Get dummy data
df_dummy = pd.get_dummies(df_model)
df_dummy[df_dummy.select_dtypes(include=['bool']).columns] = df_dummy.select_dtypes(include=['bool']).astype(int)
# Train/Test split
from sklearn.model_selection import train_test_split
X = df_dummy.drop('avg_salary', axis=1)
y = df_dummy['avg_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
import statsmodels.api as sm 
X_sm = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 

lr = LinearRegression()
lr.fit(X_train, y_train)

cv_score_lr = cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print(np.mean(cv_score_lr))

# Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

cv_score_lasso = cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print(np.mean(cv_score_lasso))
alphas = []
errors = []
for i in range(100):
    alphas.append(i/1000)
    lasso = Lasso(alpha=alphas[i])
    error = np.mean(cross_val_score(lasso, X_train, y_train, scoring="neg_mean_absolute_error",cv=5))
    errors.append(error)
alphas_errors_dict = dict(zip(alphas, errors))
def min_of_dict(d):
    best = 0 
    mini = max(d.values())
    for k, v in d.items():
        if  d[k] == mini:
            best =  k
    return best

best_alpha = min_of_dict(alphas_errors_dict)
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train, y_train)
cv_score_lasso1 = np.mean(cross_val_score(lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=10))
print(best_alpha, cv_score_lasso1)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring="neg_mean_absolute_error", cv=10))
# Tune models GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'n_estimators': range(10, 300, 100),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['squared_error', 'absolute_error']
}
gs_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="neg_mean_absolute_error")
gs_rf.fit(X_train, y_train)
gs_rf.best_params_
gs_final = gs_rf.best_estimator_

# Test ensembles
preds_lr = lr.predict(X_test) 
preds_lasso = final_lasso.predict(X_test)
preds_rf = gs_final.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, preds_lr))
print(mean_absolute_error(y_test, preds_lasso))
print(mean_absolute_error(y_test, preds_rf))

print(mean_absolute_error(y_test, preds_lasso+preds_rf)/2)