import pymongo, random, os, shutil
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model as lm
from sklearn import neighbors as nei
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn import metrics

def hyperOptimizationTest(regressor,regressorLabel,X,y,parameter_space,scoring):
    if isinstance(scoring, list):
        clf = GridSearchCV(regressor(), parameter_space, n_jobs=-1, cv=10,verbose=1,scoring=scoring,refit=scoring[0])
        clf.fit(X, y)
        print('Best '+str(regressorLabel)+': %0.3f (+/-%0.03f) for %r\n' % (clf.cv_results_['mean_test_'+str(scoring[0])][clf.best_index_],clf.cv_results_['std_test_'+str(scoring[0])][clf.best_index_]*2,clf.cv_results_['params'][clf.best_index_]))
        if (not os.path.exists('results/Bests'+'_'.join(scoring))) or (not os.path.getsize('results/Bests'+'_'.join(scoring)) > 0):
            with open('results/Bests'+'_'.join(scoring),'w') as bestFile:
                headerParts = ['algorithm']
                headerParts = headerParts + ['meansScore_'+score for score in scoring]
                headerParts = headerParts + ['stdsScore_'+score for score in scoring]
                headerParts = headerParts + ['meanFitTime','stdFitTime','meanScoreTime','stdScoreTime','params']
                bestFile.write(','.join(headerParts)+'\n')
        with open('results/Bests'+'_'.join(scoring),'a+') as bestFile:
            dataParts = [regressorLabel]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_test_'+score][clf.best_index_] for score in scoring]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_test_'+score][clf.best_index_] for score in scoring]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_fit_time'][clf.best_index_]]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_fit_time'][clf.best_index_]]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_score_time'][clf.best_index_]]
            dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_score_time'][clf.best_index_]]
            dataParts = dataParts + ['"'+str(clf.cv_results_['params'][clf.best_index_])+'"']
            bestFile.write(','.join(dataParts)+'\n')
        with open('scores/scores_'+regressorLabel+'_'+'_'.join(scoring),'w') as outFile:
            headerParts = ['algorithm']
            headerParts = headerParts + ['meansScore_'+score for score in scoring]
            headerParts = headerParts + ['stdsScore_'+score for score in scoring]
            headerParts = headerParts + ['meanFitTime','stdFitTime','meanScoreTime','stdScoreTime','params']
            outFile.write(','.join(headerParts)+'\n')
            for i in range(len(clf.cv_results_['mean_test_'+scoring[0]])):
                dataParts = [regressorLabel]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_test_'+score][i] for score in scoring]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_test_'+score][i] for score in scoring]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_fit_time'][i]]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_fit_time'][i]]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['mean_score_time'][i]]
                dataParts = dataParts + ['%0.3f' % clf.cv_results_['std_score_time'][i]]
                dataParts = dataParts + ['"'+str(clf.cv_results_['params'][i])+'"']
                outFile.write(','.join(dataParts)+'\n')
            
    else:
        clf = GridSearchCV(regressor(), parameter_space, n_jobs=-1, cv=10,verbose=1,scoring=scoring)
        clf.fit(X, y)
        print('Best '+str(regressorLabel)+': %0.3f (+/-%0.03f) for %r\n' % (clf.cv_results_['mean_test_score'][clf.best_index_],clf.cv_results_['std_test_score'][clf.best_index_]*2,clf.cv_results_['params'][clf.best_index_]))
        meansScore = clf.cv_results_['mean_test_score']
        stdsScore = clf.cv_results_['std_test_score']
        meansFitTime = clf.cv_results_['mean_fit_time']
        stdsFitTime = clf.cv_results_['std_fit_time']
        meansScoreTime = clf.cv_results_['mean_score_time']
        stdsScoreTime = clf.cv_results_['std_score_time']
        if scoring is None: scoring = ""
        with open('results/Bests'+str(scoring),'a+') as bestFile:
            bestFile.write('#s,%0.3f,%0.03f,%0.03f,%0.03f,%0.03f,%0.03f,%r\n' % (meansScore[clf.best_index_],stdsScore[clf.best_index_]*2,meansFitTime[clf.best_index_],stdsFitTime[clf.best_index_]*2,meansScoreTime[clf.best_index_],stdsScoreTime[clf.best_index_]*2,clf.cv_results_['params'][clf.best_index_]))
        with open('scores/scores_'+regressorLabel+'_'+str(scoring),'w') as outFile:
            for i in range(len(meansScore)): 
                outFile.write('#s,%0.3f,%0.03f,%0.03f,%0.03f,%0.03f,%0.03f,"%r"\n' % (meansScore[i],stdsScore[i]*2,meansFitTime[i],stdsFitTime[i]*2,meansScoreTime[i],stdsScoreTime[i]*2,clf.cv_results_['params'][i]))
    
    

#Train and test models
def testMLP(X,y,scoring):
    parameter_space = {
        'hidden_layer_sizes': [
            (int(X.shape[1]/2),int(X.shape[1]/4)),
            (int(X.shape[1]/4),int(X.shape[1]/8)),
            (int(X.shape[1]/8),int(X.shape[1]/16)),
            (int(X.shape[1]/2),),
            (int(X.shape[1]/4),),
            (int(X.shape[1]/8),)
        ],
        'random_state':[1],
        'max_iter':[2000],
        'activation': ['relu'],
        'solver': ['adam'],
        'learning_rate': ['constant','adaptive'],
    }
    hyperOptimizationTest(MLPRegressor,'MLP',X,y,parameter_space,scoring)

def testSVM(X,y,scoring):
    parameter_space = {
        'epsilon':[1/(10**i) for i in range(0,12)],
        'C':[10**i for i in range(0,12)]
    }
    hyperOptimizationTest(svm.SVR,'SVM',X,y,parameter_space,scoring)
    
def testSGD(X,y,scoring):
    parameter_space = {
        'early_stopping':[True],
        'random_state':[1],
        'loss':['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
        'penalty':['l2','l1','elasticnet'],
        'alpha':[1/(10**i) for i in range(3,7)],
        'fit_intercept':[True,False],
        'learning_rate':['constant','optimal','invscaling','adaptive'],
        'epsilon':[1/(10**i) for i in range(0,3)]
    }
    hyperOptimizationTest(lm.SGDRegressor,'SGD',X,y,parameter_space,scoring)
    
def testLinearRegressor(X,y,scoring):
    parameter_space = {
        'normalize':[True,False],
        'fit_intercept':[True,False]
    }
    hyperOptimizationTest(lm.LinearRegression,'Linear',X,y,parameter_space,scoring)
        
def testRidge(X,y,scoring):
    parameter_space = {
        'random_state':[1],
        'alpha':[0.01,0.1,0.2,0.5,1],
        'fit_intercept':[True,False],
        'normalize':[True,False],
        'solver':['auto','svd','cholesky','sparse_cg','lsqr','sag']
    }
    hyperOptimizationTest(lm.Ridge,'Ridge',X,y,parameter_space,scoring)
           
def testLars(X,y,scoring):
    parameter_space = {
        'random_state':[1],
        'eps':[0.01,0.05,0.1,0.2,0.5,0.8,1.0],
        'fit_intercept':[True,False],
        'normalize':[True,False]
    }
    hyperOptimizationTest(lm.Lars,'Lars',X,y,parameter_space,scoring)
           
def testLasso(X,y,scoring):
    parameter_space = {
        'random_state':[1],
        'max_iter':[100000],
        'alpha':[0.01,0.1,0.2,0.5,1],
        'fit_intercept':[True,False],
        'normalize':[True,False],
        'selection':['cyclic','random']
    }
    hyperOptimizationTest(lm.Lasso,'Lasso',X,y,parameter_space,scoring)

def testLassoLars(X,y,scoring):
    parameter_space = {
        'random_state':[1],
        'max_iter':[100000],
        'alpha':[0.001,0.005,0.01,0.1,0.12,0.14,0.15,0.16,0.18,0.1,0.2,0.5,1],
        'eps':[0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.005,0.01],
        'fit_intercept':[True,False],
        'normalize':[True,False]
    }
    hyperOptimizationTest(lm.LassoLars,'LassoLars',X,y,parameter_space,scoring)
            
def testOMP(X,y,scoring):
    parameter_space = {
        'fit_intercept':[True,False],
        'normalize':[True,False]
    }
    hyperOptimizationTest(lm.OrthogonalMatchingPursuit,'OMP',X,y,parameter_space,scoring)
            
def testBayesARD(X,y,scoring):
    parameter_space = {
        'alpha_1':[1/(10**7)],
        'alpha_2':[1/(10**7)],
        'lambda_1':[1/(10**i) for i in range(6,8)],
        'lambda_2':[1/(10**i) for i in range(6,8)],
        'fit_intercept':[True,False],
        'normalize':[True,False]
    }
    hyperOptimizationTest(lm.ARDRegression,'BayesARD',X,y,parameter_space,scoring)
            
def testBayesRidge(X,y,scoring):
    parameter_space = {
        'alpha_1':[1/(10**i) for i in range(5,8)],
        'alpha_2':[1/(10**i) for i in range(5,8)],
        'lambda_1':[1/(10**i) for i in range(5,8)],
        'lambda_2':[1/(10**i) for i in range(5,8)],
        'fit_intercept':[True,False],
        'normalize':[True,False]
    }
    hyperOptimizationTest(lm.BayesianRidge,'BayesRidge',X,y,parameter_space,scoring)
            
def testKnearest(X,y,scoring):
    parameter_space = {
        'n_neighbors':[i for i in range(3,10)],
        'weights':['uniform','distance'],
        'algorithm':['auto','ball_tree','kd_tree','brute'],
        'leaf_size':[i*10 for i in range(1,7)],
        'p':[1,2]
    }
    hyperOptimizationTest(nei.KNeighborsRegressor,'KNearest',X,y,parameter_space,scoring)
            
def testDecisionTree(X,y,scoring):
    parameter_space = {
        'criterion':['mse','friedman_mse','mae'],
        'splitter':['best','random']
    }
    hyperOptimizationTest(DecisionTreeRegressor,'DecisionTree',X,y,parameter_space,scoring)

def runAnalysis(X,y): 
    #Test all models for best hyper-parameter configuration
    print()
    print('Testing algorithms...')
    tests = [testMLP,testSGD,testLinearRegressor,testRidge,testLars,testLasso,testLassoLars,testOMP,testBayesARD,testBayesRidge,testKnearest,testDecisionTree]
    fullScores = ['r2','neg_mean_absolute_error','neg_median_absolute_error','max_error','explained_variance']
    for test in tests:
        try:
            test(X,y,fullScores)
        except Exception as ex:
            print(str(test.__name__)+' failed.')
            print(ex)
    print('Testing Done.')
    print()
    
def getCoefs(X,regressor):
    print('Getting coefs for regressor '+str(type(regressor).__name__)+'...\n')
    X_train=X
    x = X_train.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled,columns=X_train.columns)
    coef = pd.Series(regressor.coef_, index = X.columns)
    print(str(type(regressor).__name__)+ " picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature coefficiency using "+str(type(regressor).__name__)+" Model")
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    fig.savefig('./figs/'+str(regressor)+".jpg",dpi=200)
    plt.show()
    plt.clf()
    return coef

def getImportance(X, y, regressor):
    print('Getting feature importance for regressor '+str(type(regressor).__name__)+'...\n')
    X_train=X
    x = X_train.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled,columns=X_train.columns)
    results = permutation_importance(regressor, X, y, scoring='r2')
    importance = preprocessing.normalize(X, norm='max')
    importance=pd.Series(results.importances_mean, index = X.columns)
    imp_coef = importance.sort_values()
    rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using "+str(type(regressor).__name__)+" Model")
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    fig.savefig('./figs/'+str(regressor)+"fi.jpg",dpi=200)
    plt.show()
    plt.clf()
    return importance

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