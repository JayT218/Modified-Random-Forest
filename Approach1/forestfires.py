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
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



import sys
print(sys.getrecursionlimit())

sys.setrecursionlimit(1500)

print(sys.getrecursionlimit())

filename = "/Users/jay/Desktop/Virginia Tech/Degree project/Project/forestfiresing.csv"
df = pd.read_csv(filename) # read an csv spreadsheet 
print('File ', filename, ' is of size ', df.shape)
labels = df.columns
featureLabels = labels.drop('area').values
xFrame = df[featureLabels]
yFrame = df['area']



X=xFrame
y=yFrame

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

'''print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))'''

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



N_forest = 50
rfc = [None] * N_forest
y_predict = [None] * N_forest

# bulid N forests with 1 tree
def GrowForest(N_forest, fid):
    if (N_forest != 1):
        #rfc = [None] * N_forest
        #y_predict = [None] * N_forest

        for i in range(N_forest):
            rfc[i] = RandomForestRegressor(n_estimators = 1, max_depth = 5)
            rfc[i].fit(X_train,y_train)
            y_predict[i] = rfc[i].predict(X_test)
            #print("y_predict[i]: ", y_predict[i])
        return y_predict
    else:
        rfc[fid] = RandomForestRegressor(n_estimators = 1, max_depth = 5)
        rfc[fid].fit(X_train,y_train)
        y_predict[fid] = rfc[fid].predict(X_test)
        #print("y_predict[fid]: ", y_predict[fid])
        return y_predict



y_predict = GrowForest(N_forest, N_forest)


y_test_list = y_test.tolist()

'''print("y_test: ", y_test)
print("y_test_list: ", y_test_list)
print("y_predict: ", y_predict)'''



#score_list = []
# find least accurate forest
def worst_forest():
    score_list = []

    for forest in range(N_forest):
        score = 0
        for element in range(len(y_test_list)):
            '''if y_predict[forest][element] == y_test_list[element]:
                score += 1'''
            score += (y_predict[forest][element] - y_test_list[element]) ** 2

        score_list.append(score / len(y_test_list))
    #print(score_list)

    least_accurate_forest = score_list.index(max(score_list))
    least_accurate_forest_score = max(score_list)
    #print("least_accurate_forest: ", least_accurate_forest)
    #print("least_accurate_forest_score: ", least_accurate_forest_score)
    return least_accurate_forest, least_accurate_forest_score

#least_accurate_forest = worst_forest()



# update forest
def update_forest(fid, fid_score):

    #print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])
    y_predict = GrowForest(1, fid)
    #print("y_predict[least_accurate_forest]: ", y_predict[least_accurate_forest])

    cnt = 0
    score = 0
    for element in range(len(y_test_list)):
        '''if y_predict[fid][element] == y_test_list[element]:
            cnt += 1'''
        score += (y_predict[fid][element] - y_test_list[element]) ** 2
    #score = cnt / len(y_test_list)

    #print("new forest score: ", score)
    #print("fid_score: ", fid_score)
    score = score / len(y_test_list)

    if (score > fid_score):
        update = 1
    else:
        update = 0
    return update

    '''while(score < fid_score or loop > 0):
        loop -= 1
        print("loop: ", loop)
        update_forest(fid, fid_score)
    return'''






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
        #dicvote = {}
        count = 0
        for row in range(N_forest):
            '''if y_predict[row][column] not in dicvote:
                dicvote[y_predict[row][column]] = 1
            else:
                dicvote[y_predict[row][column]] += 1
            

        dicvotekey_list = list(dicvote.keys())
        dicvoteval_list = list(dicvote.values())
        #dicvoteval_list.sort(reverse = True)
        count = dicvoteval_list.index(max(dicvoteval_list))
        vote.append(dicvotekey_list[count])'''
            count += y_predict[row][column]
        count = count / N_forest
        vote.append(count)



    #print("vote: ", vote)
    return vote

#vote = voting()


# calculate accuracy
def accuracy():
    count = 0
    for i in range(len(y_test_list)):
        '''if vote[i] == y_test_list[i]:
            count += 1'''
        count += (vote[i] - y_test_list[i]) ** 2
    count = count / len(y_test_list)
    print("MSE: ", count)
    #print("Root Mean Squared Error: ", np.sqrt(count))

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


for i in range(50):
    least_accurate_forest, least_accurate_forest_score = worst_forest()
    vote = voting()
    accuracy()
    update = 1
    while(update == 1):
        update = update_forest(least_accurate_forest, least_accurate_forest_score)
    #update = update_forest(least_accurate_forest, least_accurate_forest_score)



regressor = RandomForestRegressor(n_estimators=50, max_depth=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#print("y_pred: ", y_pred)
print('SKlearnMSE:', metrics.mean_squared_error(y_test, y_pred))



#rfc.fit(X_train,y_train)

#y_predict=rfc.predict(X_test)




    			 

#print("rfc.score(X_test,y_test): ", rfc.score(X_test,y_test))