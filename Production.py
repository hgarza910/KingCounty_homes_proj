import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import models

df = pd.read_csv('kc_house_data_cleaned.csv')

# choose columns to use
df_model = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement',
               'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']]
#%%
X = df_model.drop('price', axis = 1)
y = df_model.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%

xg_model = models.xg_tune(X_train, X_test, y_train, y_test, XGBRegressor)
#%%
