import pymongo, json, csv, traceback, sys, os

myclient = pymongo.MongoClient("mongodb://localhost/Corn2020")
mydb = myclient["Maize2020"]
corncol = mydb["Corn"]