from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.impute import KNNImputer

from sklearn import datasets

import sys
print(sys.getrecursionlimit())

sys.setrecursionlimit(1500)

print(sys.getrecursionlimit())

'''filename = "/Users/jay/Desktop/Virginia Tech/Degree project/Project/adult.csv"
df = pd.read_csv(filename) # read an csv spreadsheet 
print('File ', filename, ' is of size ', df.shape)
labels = df.columns
print(labels)
featureLabels = labels.drop('Income').values
xFrame = df[featureLabels]
yFrame = df['Income']

df_dumx=pd.get_dummies(xFrame)
#print(df_dumx)

df_dumy=pd.get_dummies(yFrame)
#print(df_dumy)


imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
imputer.fit(df_dumx)
X = imputer.transform(df_dumx)
y = df_dumy'''

iris=datasets.load_iris()
#print("iris: ", iris)


X=iris.data
y=iris.target



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

'''print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))'''


N_forest = 5
rfc = [None] * N_forest
y_predict = [None] * N_forest

# bulid N forests with 1 tree
def GrowForest(New_forest, ids):
    if (New_forest == N_forest):
        #rfc = [None] * N_forest
        #y_predict = [None] * N_forest

        for i in range(New_forest):
            rfc[i] = RandomForestClassifier(n_estimators = 1, max_depth = 5)
            rfc[i].fit(X_train,y_train)
            y_predict[i] = rfc[i].predict(X_test)
            #print("y_predict[i]: ", y_predict[i])
        return y_predict
    else:
        for fid in ids:
            rfc[fid] = RandomForestClassifier(n_estimators = 1, max_depth = 5)
            rfc[fid].fit(X_train,y_train)
            y_predict[fid] = rfc[fid].predict(X_test)
            #print("y_predict[fid]: ", y_predict[fid])
        return y_predict



y_predict = GrowForest(N_forest, N_forest)


y_test_list = y_test.tolist()

'''print("y_test: ", y_test)
print("y_test_list: ", y_test_list)
print("y_predict[0]: ", y_predict[0])'''




#score_list = []
# find least accurate forest
def worst_forest():
    score_list = []

    for forest in range(N_forest):
        score = 0
        for element in range(len(y_test_list)):
            if y_predict[forest][element] == y_test_list[element]:
                score += 1
        score_list.append(score / len(y_test_list))
    #print(score_list)

    least_accurate_forest = score_list.index(min(score_list))
    least_accurate_forest_score = min(score_list)
    #print("least_accurate_forest: ", least_accurate_forest)
    #print("least_accurate_forest_score: ", least_accurate_forest_score)
    return least_accurate_forest, least_accurate_forest_score






import statistics
def bad_forests(current_level):
    score_list = []
    victim_forests = []
    victim_forests_score = []
    victim_forests_score_avg = 0

    for forest in range(N_forest):
        score = 0
        for element in range(len(y_test_list)):
            if y_predict[forest][element] == y_test_list[element]:
                score += 1
        score = score / len(y_test_list)
        score_list.append(score)
        if score < current_level:
            victim_forests.append(forest)
            victim_forests_score.append(score)
    
    '''print("score_list: ", score_list)
    print("victim_forests: ", victim_forests)
    print("victim_forests_score: ", victim_forests_score)'''

    if victim_forests_score:
        victim_forests_score_avg = statistics.mean(victim_forests_score)
        #print("victim_forests_score_avg: ", victim_forests_score_avg)

    #least_accurate_forest = score_list.index(min(score_list))
    #least_accurate_forest_score = min(score_list)
    #print("least_accurate_forest: ", least_accurate_forest)
    #print("least_accurate_forest_score: ", least_accurate_forest_score)
    #print("bad_forests: ", current_level)
    #return least_accurate_forest, least_accurate_forest_score
    return victim_forests, victim_forests_score_avg










#least_accurate_forest = worst_forest()



# update forest
def update_forest(fid, fid_score):

    #print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])
    y_predict = GrowForest(len(fid), fid)
    #print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])

    score = 0
    for i in fid:

        cnt = 0
        for element in range(len(y_test_list)):
            if y_predict[i][element] == y_test_list[element]:
                cnt += 1
        score += cnt / len(y_test_list)

    #print("new forest score: ", score)
    #print("fid_score: ", fid_score)
    
    score = score / len(fid)



    if (score < fid_score):
        update = 1
    else:
        update = 0
    return update







#print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])
#update_forest()
#print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])




'''vote = []
count = 0'''

vote = []

# voting process
def voting():

    vote = []
    count = 0


    for column in range(len(y_test_list)):
        dicvote = {}
        for row in range(N_forest):
            #print("y_predict[row][column]: ", y_predict[row][column])
            #print("y_predict[row][column][0]: ", y_predict[row][column][0])
            if y_predict[row][column] not in dicvote:
                dicvote[y_predict[row][column]] = 1
            else:
                dicvote[y_predict[row][column]] += 1

        dicvotekey_list = list(dicvote.keys())
        dicvoteval_list = list(dicvote.values())
        #dicvoteval_list.sort(reverse = True)
        count = dicvoteval_list.index(max(dicvoteval_list))
        vote.append(dicvotekey_list[count])
    #print("vote: ", vote)
    return vote

#vote = voting()


accurate_value = 0
# calculate accuracy
def accuracy():
    count = 0
    for i in range(len(y_test_list)):
        if vote[i] == y_test_list[i]:
            count += 1
    print("accuracy: ", count/len(y_test))
    accurate_value = count/len(y_test)
    return accurate_value

    #print('R2 = %6.4f,' % r2_score(y_test_list, vote), end = '', flush = True)

    #print(' MSE = %6.4f,' % mean_squared_error(y_test_list, vote), end = '', flush = True)


    #print(' AUROC = %6.4f,' % roc_auc_score(y_test_list, vote), end = '', flush = True)


    #print(' Precision = %6.4f,' % precision_score(y_test_list, vote), end = '', flush = True)


    #print(' Recall = %6.4f,' % recall_score(y_test_list, vote), end = '', flush = True)

    '''print('Confusion_matrix: ')
    print(confusion_matrix(y_test_list, vote))'''



    #return count
    
#accuracy()


#print("accuracy: ", right/len(y_test))


for i in range(5):
    #least_accurate_forest, least_accurate_forest_score = worst_forest()
    vote = voting()
    accurate_value = accuracy()
    #print("accurate_value: ", accurate_value)
    victim_forests, victim_forests_score_avg = bad_forests(accurate_value)
    #print("bad_forests: ", victim_forests)
    if len(victim_forests) == 0:
        break
    update = 1
    while(update == 1):
        update = update_forest(victim_forests, victim_forests_score_avg)



#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

'''clf = RandomForestClassifier(n_estimators = 50, max_depth = 5).fit(X_train, y_train)

print('Accuracy of RF classifier on test set: ',clf.score(X_test, y_test))
'''


#rfc.fit(X_train,y_train)

#y_predict=rfc.predict(X_test)




    			 

#print("rfc.score(X_test,y_test): ", rfc.score(X_test,y_test))