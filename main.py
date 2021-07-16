import pymongo, json, csv, traceback, sys, os
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

    
def transformVectors(vectors,dataset='full'):
    newVectors = {}
    allKeys = {}
    for vector in vectors:
        for key in vector.keys():
            allKeys[key] = 1
    newVectors = {key:[] for key in allKeys}
    categoricals = []
    for vector in vectors:
        for key in allKeys:
            if key in vector and isinstance(vector[key], str):
                if key not in categoricals: 
                    categoricals.append(key)
    for vector in vectors:
        canceled = False
        for key in categoricals:
            if key not in vector or str(vector[key]).strip() in ['',' ','-']:
                canceled = True
                break
                
        if canceled:
            print(vector)
        else:
            for key in allKeys:
                if key in vector:
                    newVectors[key].append(vector[key])
                else:
                    newVectors[key].append(NaN)
    for key in categoricals:
        labelEncoders[key] = preprocessing.LabelEncoder()
        labelEncoders[key].fit(newVectors[key])
        with open('./encodings/'+str(key)+'_'+str(dataset)+'.encoding','w') as enc_file:
            enc_file.write(str(list(labelEncoders[key].classes_))+'\n')
            enc_file.write(str(list(labelEncoders[key].transform(list(labelEncoders[key].classes_))))+'\n')
        newVectors[key] = labelEncoders[key].transform(newVectors[key])
        

    
    return pd.DataFrame.from_dict(newVectors)

def prepareData(excludedCols = []):
    corn_Projection = {'$project':{'_id':0}}
    for excluded in excludedCols:
        corn_Projection['$project'][excluded] = 0
    
    corn_vect = list(corncol.aggregate([corn_Projection]))
    #Create data frame
    totalFrame = transformVectors(corn_vect)    
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


def getCoefs(X,regressor):
    print('Getting coefs for regressor '+str(type(regressor).__name__)+'...\n')
    coef = pd.Series(regressor.coef_, index = X.columns)
    print(str(type(regressor).__name__)+ " picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using "+str(type(regressor).__name__)+" Model")
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    fig.savefig('./figs/feature_coefs.jpg',dpi=200)
    plt.show()
    plt.clf()
    return coef

def getImportance(X, y, regressor):
    print('Getting feature importance for regressor '+str(type(regressor).__name__)+'...\n')
    results = permutation_importance(regressor, X, y, scoring='r2')
    importance = pd.Series(results.importances_mean, index = X.columns)
    imp_coef = importance.sort_values()
    rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using "+str(type(regressor).__name__)+" Model")
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    fig.savefig('./figs/feature_importance.jpg',dpi=200)
    plt.show()
    plt.clf()
    return importance



dataset = prepareData()

print('Starting analysis for '+str(dataset)+'...')
X,y,totalFrame = dataset
regressor = nei.KNeighborsRegressor(algorithm='ball_tree', leaf_size=10, n_neighbors=9, p=1, weights='distance')
regressor.fit(X,y)
print(getImportance(X, y, regressor).sort_values(ascending=False))
print()
print()
regressor2 = lm.LassoLars(alpha=0.01, eps=1e-11, fit_intercept=True, max_iter=100000, normalize=False, random_state=1)
regressor2.fit(X,y)
print(getCoefs(X,regressor2).sort_values(ascending=False))
