import pymongo, json, csv, traceback, sys, os,algorithms
from config import corncol, mydb
import json, csv
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import neighbors as nei
from sklearn import linear_model as lm
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn import metrics



def CornParser(filename="SweerCorn2020.csv"):
    results = []
    try:
        with open(filename) as in_file:
            in_reader = csv.reader(in_file, delimiter=';')
            headers = next(in_reader, None)
            headers[0] = 'yield'
            headers = [header.strip() for header in headers]
            result = {}
            i = 0
            try:
                corncol.drop()
            except:
                None
            for row in in_reader:
                result['_id'] = i
                i += 1
                for datumIndex, header in enumerate(headers):
                    try:
                        datum = row[datumIndex].strip()
                        try:
                            if '.' not in datum and ',' in datum:
                                datum = datum.replace(',','.').replace('..','.')
                            while '..' in datum:
                                datum = datum.replace('..','.')
                            result[header] = float(datum)
                        except:
                            result[header] = datum.upper().strip()
                    except: 
                        traceback.print_exc(file=sys.stdout)
                        continue
                try:
                    print((corncol.insert_one(result)).inserted_id)
                except:
                    traceback.print_exc(file=sys.stdout)
                results.append(result)
    except:
        traceback.print_exc(file=sys.stdout)
    return results

    


def prepareData(excludedCols = []):
    corn_Projection = {'$project':{'_id':0}}
    for excluded in excludedCols:
        corn_Projection['$project'][excluded] = 0    
    corn_vect = list(corncol.aggregate([corn_Projection]))
    #Create data frame
    totalFrame = algorithms.transformVectors(corn_vect)    
    #Clean NaN values and create label and features sets
    y = totalFrame.loc[:, 'yield']
    X = totalFrame.loc[:, sorted([col for col in totalFrame.columns if col not in ['yield']])]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    trainX = imp.transform(X)
    trainXDataframe = pd.DataFrame(trainX)
    trainXDataframe.columns = X.columns
    trainXDataframe.index=X.index
    X = trainXDataframe.loc[:, sorted([col for col in trainXDataframe.columns if col not in ['yield']])]
    print('totalFeatures',X.shape)
    print()
    print(totalFrame.head)
    print()
    print(X.head)
    print()
    print(y.head)
    return (X,y,totalFrame)





dataset = prepareData()
print('Starting analysis for '+str(dataset)+'...')
X,y,totalFrame = dataset
#algorithms.runAnalysis(X,y)
regressor = nei.KNeighborsRegressor(algorithm='ball_tree', leaf_size=10, n_neighbors=9, p=1, weights='distance')
regressor.fit(X,y)
print(algorithms.getImportance(X, y, regressor).sort_values(ascending=False))
print()
print()
regressor2 = lm.LassoLars(alpha=0.01, eps=1e-11, fit_intercept=True, max_iter=100000, normalize=False, random_state=1)
regressor2.fit(X,y)
print(algorithms.getCoefs(X,regressor2).sort_values(ascending=False))
