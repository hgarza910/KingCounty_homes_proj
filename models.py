


#  file serves as the hub for all the models for testing

# OLS Regression
def ols_reg(X, y):
    # ols need constant
    X_sm = X = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    return model

# Linear Regression
def lin_reg(X_train, y_train, LinearRegression):
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    return model

# LGBM Regression
def lgbm_reg(X_train, y_train, LGBMRegressor):
    model = LGBMRegressor()
    model = model.fit(X_train, y_train)
    return model

# Lasso Regression
def lasso_reg(X_train, y_train, Lasso):
    model = Lasso(tol=.1)
    model = model.fit(X_train, y_train)
    return model

# Random Forest Regression
def rf_reg(X_train, y_train, RandomForestRegressor):
    model = RandomForestRegressor()
    model = model.fit(X_train, y_train)
    return model

# XGboost Regression
def xgboost_reg(X_train, y_train, XGBRegressor):
    model = XGBRegressor()
    model = model.fit(X_train, y_train)
    return model

# CatBoost Regression
def cat_reg(X_train, y_train, CatBoostRegressor):
    model = CatBoostRegressor()
    model = model.fit(X_train, y_train)
    return model

# SGDRegressor
def sgd_reg(X_train, y_train, SGDRegressor):
    model = SGDRegressor()
    model = model.fit(X_train, y_train)
    return model

# Kernal Ridge Regression
def kern_reg(X_train, y_train, KernelRidge):
    model = KernelRidge()
    model = model.fit(X_train, y_train)
    return model

# Elastic Net Regression
def elast_reg(X_train, y_train, ElasticNet):
    model = ElasticNet()
    model = model.fit(X_train, y_train)
    return model

# Bayesian Ridge Regression
def bayesr_reg(X_train, y_train, BayesianRidge):
    model = BayesianRidge()
    model = model.fit(X_train, y_train)
    return model

# Gradient boost Regression
def gb_reg(X_train, y_train, GradientBoostingRegressor):
    model = GradientBoostingRegressor()
    model = model.fit(X_train, y_train)
    return model

# Support Vector Machine Regression
def svm_reg(X_train, y_train, SVR):
    model = SVR()
    model = model.fit(X_train, y_train)
    return model



# Classification models

# Naive Bayes-Gaussian
def nbg_class(X_train, y_train, GaussianNB):
    model = GaussianNB()
    model = model.fit(X_train, y_train)
    return model

#  Naive Bayes-Multinomial
def nbm_class(X_train, y_train, MultinomialNB):
    model = MultinomialNB()
    model = model.fit(X_train, y_train)
    return model

# Logistic Regression
def log_class(X_train, y_train, LogisticRegression):
    model = LogisticRegression(random_state=42, multi_class='auto', solver='lbfgs', max_iter=1000)
    model = model.fit(X_train, y_train)
    return model

# K-nearest neighbors
def knn_class(X_train, y_train, KNeighborsClassifier):
    model = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2)
    model = model.fit(X_train, y_train)
    return model

# Support Vector Machines
def svm_class(X_train, y_train, SVC):
    model = SVC(kernel='rbf', gamma='auto')
    model = model.fit(X_train, y_train)
    return model

# Stochastic Gradient Descent
def sgd_class(X_train, y_train, SGDClassifier):
    model = SGDClassifier()
    model = model.fit(X_train, y_train)
    return model
# Decision Tree
def dt_class(X_train, y_train, DecisionTreeClassifier):
    model = DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    return model

# Gradient Boosting
def gb_class(X_train, y_train, GradientBoostingClassifier):
    model = GradientBoostingClassifier()
    model = model.fit(X_train, y_train)
    return model

# LGBM
def lgbm_class(X_train, y_train, LGBMClassifier):
    model = LGBMClassifier()
    model = model.fit(X_train, y_train)
    return model

# XGBoost
def xgboost_class(X_train, y_train, XGBClassifier):
    model = XGBClassifier(objective='binary:logistic', random_state=42)
    model = model.fit(X_train, y_train)
    return model

# Random Forest
def rf_class(X_train, y_train, RandomForestClassifier):
    model = RandomForestClassifier()
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
                      early_stopping_rounds=5,
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
def test_models(models):
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    print(predictions)

#%%
