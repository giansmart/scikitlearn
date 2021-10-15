#Randomized Search CV (Cross Validation) Method
import pandas as pd
from pandas.io.sql import DatabaseError

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.head(5))

    X = dataset.drop(['country','rank','score'], axis=1)
    y = dataset['score']

    #definir el regresor
    reg = RandomForestRegressor()

    #grilla de parametros
    parametros = {
        'n_estimators': range(4,16), #entre 4 y 15
        'criterion': ['mse','mae'], #medidas de calidad
        'max_depth': range(2,11)
    }

    rand_est = RandomizedSearchCV(
        reg, 
        parametros, 
        n_iter=10, #10 diferentes configuraciones de parametros
        cv=3, #cant pliegues (2 de training y 1 de test)
        scoring="neg_mean_absolute_error"
    ).fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))