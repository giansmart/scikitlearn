import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #sirve para clasificacion

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(["target"],axis=1, inplace=False) #inplace=True para que afecte el dataframe original (elimine la columna)
    y = dt_heart['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_predict = knn_class.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_predict, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))