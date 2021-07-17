from config import corncol
import algorithms
import pingouin as pg
import pandas as pd
import csv

cornProjection = {'$project':{'_id':0}}
cornvectors = list(corncol.aggregate([cornProjection]))
#Create data frame
totalFrame = algorithms.transformVectors(cornvectors)
variables =[]
names=[]
for col in totalFrame.columns:
    col=totalFrame.loc[:,col]
    variables.append(col)
    names.append(col.name)

def complex_algorithms_list(x):
    data = []
    for i, var in enumerate(variables):
        data+=list((pg.corr(variables[0], variables[i], method=x)[['r','r2','adj_r2']]).itertuples(index=False))
    df = pd.DataFrame.from_records(data, columns=['r','r2','adj_r2'],index=names)
    df.to_csv("Complex_results_"+x+".csv")
    print(df)


def run_complex():
    list=['kendall','bicor','percbend','shepherd','skipped']
    for l in list:
        print("running %s" %l)
        complex_algorithms_list(l)

run_complex()