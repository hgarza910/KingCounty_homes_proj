
#  file serves as the hub for all the models for testing
#%%
# OLS Regression
def ols_reg(X, y, sm):
    # ols need constant
    X_sm = X = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    #result = model.fit().summary()
    return model
#%%
# Linear Regression
def lin_reg(X_train, y_train, LinearRegression):
    model = LinearRegression()
    model_fit = model.fit(X_train, y_train)
    return model_fit

# Lasso Regression
def lasso_reg(X_train, y_train, Lasso):
    model = Lasso(tol=.1)
    model = model.fit(X_train, y_train)
    return model

# Random Forest Regression
def rand_forest(X_train, y_train, RandomForestRegressor):
    model = RandomForestRegressor()
    model = model.fit(X_train, y_train)
    return model

def xgboost_reg(X_train, y_train, XGBRegressor):
    model = XGBRegressor()
    model = model.fit(X_train, y_train)
    return model


# Tuning

def gridsearch(est, X_train, y_train, GridSearchCV):
    parameters = {'n_estimators':range(10,100,10)}
    model = GridSearchCV(est, parameters, scoring='neg_mean_absolute_error')
    model = model.fit(X_train, y_train)
    return model

def xg_tune(X_train, X_test, y_train, y_test, XGBRegressor):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    model = model.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    return model

def get_means(models):
    means = []
    for model in models:
        mean = np.mean(cross_val_score(model, X_train, y_train, scoring= 'neg_mean_absolute_error'))
        means.append(mean)
    print(means)

#test models
def test_models(model, X_test, y_test, accuracy_score):
    prediction = model.predict(X_test)
    #predictions = [round(value) for value in prediction]
    #print(predictions)
    #accuracy = accuracy_score(y_test, predictions)
    #print(accuracy)
    return prediction
    #return accuracy
    #print("Accuracy: %.2f%%" % (accuracy * 100))
    #print(predictions)





