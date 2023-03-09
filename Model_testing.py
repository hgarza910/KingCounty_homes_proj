import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import models

df = pd.read_csv('kc_house_data_cleaned.csv')

# choose columns to use
df_model = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement',
               'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']]
#%%
X = df_model.drop('price', axis = 1)
y = df_model.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mods = []
#%%
# ols regression
ols_model = models.ols_reg(X, y, sm)
ols_model.fit().summary()
#%%
# Linear Regression
linear_model = models.lin_reg(X_train, y_train, LinearRegression)
#pred_lm = linear_model.predict(X_test)
#print(pred_lm)
mods.append(linear_model)
# Lasso Regression
lasso_model = models.lasso_reg(X_train, y_train, Lasso)
mods.append(lasso_model)
# Random forest regression
rf_model = models.rand_forest(X_train, y_train, RandomForestRegressor)
mods.append(rf_model)
# XGBoost
xg_model = models.xgboost_reg(X_train, y_train, XGBRegressor)
mods.append(xg_model)
# tune XGBoost model
xg_tune_model = models.xg_tune(X_train, X_test, y_train, y_test, XGBRegressor)
mods.append(xg_tune_model)
# tune random forest model with gridsearch
rf_gs_tune = models.gridsearch(rf_model, X_train, y_train, GridSearchCV)
mods.append(rf_gs_tune)
# tune xgb model with gridsearchCV??
xg_gs_tune = models.gridsearch(xg_model, X_train, y_train, GridSearchCV)
mods.append(xg_gs_tune)


#%%
# get mae of models
maes = []
for mod in mods:
    mae = mean_absolute_error(y_test, mod.predict(X_test))
    maes.append(mae)

#%%
for mae in maes:
    print(mae)
#%%

