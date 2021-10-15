from numpy.core.fromnumeric import mean
import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv('./data/candy.csv')
    #print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1) #se elimina al ser una columna categorica que no se puede entrenar
    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_)) #para saber la cantidad de etiquetas
    print("="*62)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print(dataset)