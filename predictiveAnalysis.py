# Links:
# https://towardsdatascience.com/predicting-the-outcome-of-nba-games-with-machine-learning-a810bb768f20
# http://cs229.stanford.edu/proj2012/Wheeler-PredictingNBAPlayerPerformance.pdf
# credit from https://github.com/llSourcell/Predicting_Winning_Teams

from json import load
import numpy as np
from numpy.core.fromnumeric import choose
import pandas as pd
from datetime import datetime as dt
from statistics import mean
import itertools
import datetime
from scipy.sparse.construct import rand
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score
from sklearn.utils.sparsefuncs import mean_variance_axis

import xgboost as xgb
#the outcome (dependent variable) has only a limited number of possible values. 
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#feature selection libraries
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
import pickle
import dataScraper

def categorizeFantasyPoints(row):
    if(row['Fantasy Score']<10):
        return '< 10'
    elif(row['Fantasy Score']>=10 and row['Fantasy Score']<20):
        return '10 - 20'
    elif(row['Fantasy Score']>=20 and row['Fantasy Score']<30):
        return '20 - 30'
    elif(row['Fantasy Score']>=30 and row['Fantasy Score']<40):
        return '30 - 40'
    elif(row['Fantasy Score']>=40 and row['Fantasy Score']<50):
        return '40 - 50'
    elif(row['Fantasy Score']>=50):
        return '>50'
    else:
        return "NAN"

def categorizeFantasyPoints1(row):
    if(row['Fantasy Score']<(row["AvgFantScore"]-5)):
        return '< Fantasy Average' 
    elif(row['Fantasy Score']>(row["AvgFantScore"]+5)):
        return '> Fantasy Average'
    else:
        return "Near Fantasy Average"

def categorizeHorA(row):
    if(pd.isnull(row["H or A"])):
        return "H"
    if(row["H or A"]=="@"):
        return "A"

#time information found here: https://stackoverflow.com/questions/2780897/python-summing-up-time
def avgMPOverXGames(MP,numDays):
    avg = [0] * numDays
    for i in range(numDays,len(MP)):
        timeSum = datetime.timedelta()
        for j in range (i-numDays,i):
            (m, s) = MP[j].split(':')
            d = datetime.timedelta(minutes=int(m), seconds=int(s))
            timeSum += d
        avg.append(int((timeSum/3).seconds))
    return avg

def getAvgSoFar(col):
    col = list(col)
    avg = col[:3]
    for i in range(3,len(col)):
        avg.append(mean(col[:i]))
    return avg

def getLast3Games(df):
    numDays = 3
    lastGame,twoGamesAgo,threeGamesAgo = ([0,0,0] for i in range(3))
    fantasyPoints = list(df["Fantasy Score"])
    for i in range(numDays,len(fantasyPoints)):
        lastGame.append(fantasyPoints[i-1])
        twoGamesAgo.append(fantasyPoints[i-2])
        threeGamesAgo.append(fantasyPoints[i-3])
    df["Last Game"] = lastGame
    df["2 Games Ago"] = twoGamesAgo
    df["3 Games Ago"] =  threeGamesAgo
    return df

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    return output

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    print(y_pred)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, average="micro"), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

def getAllPlayerGameLogs(allPlayers,train):
    if(not train):
        opponents = list(allPlayers["Opp1"])
        playerNames = list(allPlayers["Name"])
        playerPos = allPlayers["Position"]
    else:
        playerNames = allPlayers["Player"]
        playerPos = allPlayers["Pos"]

    headersDf = pd.read_csv("Other/bballRefHeaders.csv")
    headers = list(headersDf["headers"])
    headersDf = pd.read_csv("Other/bballRefHeadersAdvanced.csv")
    headersAdvanced = list(headersDf["headers"])

    allPlayerData = pd.DataFrame(columns=headers)
    allPlayerDataAdvanced = pd.DataFrame(columns=headersAdvanced)

    for player in range(len(playerNames)):
        playerData = dataScraper.scrapeBballRefPlayerData(playerNames[player],train,False)
        playerDataAdvanced = dataScraper.scrapeBballRefPlayerData(playerNames[player],train,True)
        #playerData = playerData.merge(playerDataAdvanced,on = ['Date','Age','Tm'])
        #SHOULD MERGE THESE DATAFRAMES
        playerData = playerData.apply(pd.to_numeric, errors='ignore')
        playerDataAdvanced = playerDataAdvanced.apply(pd.to_numeric, errors='ignore')
        playerData = dataScraper.calculateFantasyScore(playerData)

        # if we are training, only get data for players that have played more than 40 games in the last 2y
        if((len(playerData)>40 and train) or (not train)):
            #print(playerNames[player])
            if(not train):
                # playerData = playerData.tail(3)
                # playerDataAdvanced = playerDataAdvanced.tail(3)
                if(opponents[player][0]=="@"):
                    HorA = "@"
                    opp = opponents[player][1:]
                else:
                    HorA = ''
                    opp = opponents[player]

                todaysGameRow = playerData.iloc[-1:]
                todaysGameRow["index"] = todaysGameRow.iat[0,0]
                todaysGameRow = todaysGameRow.set_index("index")
                todaysGameRow.iat[0,4] = HorA
                todaysGameRow.iat[0,5] = opp
                
                todaysGameRowAdv = playerData.iloc[-1:]
                todaysGameRowAdv["index"] = todaysGameRowAdv.iat[0,0].copy()
                todaysGameRowAdv = todaysGameRowAdv.set_index("index")
                todaysGameRowAdv.iat[0,4] = HorA
                todaysGameRowAdv.iat[0,5] = opp

                playerData = playerData.append(todaysGameRow)
                playerDataAdvanced = playerDataAdvanced.append(todaysGameRowAdv)
            if(len(playerDataAdvanced)>3):
                playerData = getLast3Games(playerData)
                playerData['Name'] = playerNames[player] 
                playerData['AvgMP'] = avgMPOverXGames(list(playerData["MP"]),3)
                playerData['AvgPTS'] = getAvgSoFar(playerData["PTS"])
                playerData['AvgREB'] = getAvgSoFar(playerData["TRB"])
                playerData['AvgAST'] = getAvgSoFar(playerData["AST"])
                playerData['AvgSTL'] = getAvgSoFar(playerData["STL"])
                playerData['AvgBLK'] = getAvgSoFar(playerData["TOV"])
                playerData['AvgFantScore'] = getAvgSoFar(playerData["Fantasy Score"])
                playerData["AvgORtg"] = getAvgSoFar(playerDataAdvanced["ORtg"])
                playerData["AvgDRtg"] = getAvgSoFar(playerDataAdvanced["DRtg"])
                playerData["AvgTS%"] = getAvgSoFar(playerDataAdvanced["TS%"])
                playerData["AvgUSG%"] = getAvgSoFar(playerDataAdvanced["USG%"])
                playerData["AvgeFG%"] = getAvgSoFar(playerDataAdvanced["eFG%"])
                playerData["AvgGmSc%"] = getAvgSoFar(playerDataAdvanced["GmSc"])
            playerData["Score Category"] = playerData.apply(lambda row: categorizeFantasyPoints(row), axis=1)
            playerData["Pos"] = playerPos[player]
            if(train):
                allPlayerData = allPlayerData.append(playerData.tail(-3))
            else:
                allPlayerData = allPlayerData.append(playerData.tail(1))
            allPlayerDataAdvanced = allPlayerDataAdvanced.append(playerDataAdvanced.tail(-3))
    #export to a csv. It will hold all game logs of all the players that are playing this season from the past 2 years
    allPlayerData["H or A"] = allPlayerData.apply(lambda row: categorizeHorA(row), axis=1)
    allPlayerData["H or A"] = allPlayerData["H or A"].fillna("H")
    if(train):
        allPlayerData.to_csv("data/AllPlayerData.csv",index=False)
        allPlayerDataAdvanced.to_csv("data/AllPlayerDataAdvanced.csv",index=False)
    return allPlayerData

#prob dont need anymore
def cleanData():
    allPlayerData = pd.read_csv("data/AllPlayerData.csv")
    allPlayerDataAdvanced = pd.read_csv("data/AllPlayerDataAdvanced.csv")
    allPlayerData["H or A"] = allPlayerData["H or A"].fillna("H")
    allPlayerData["ScoreCategory1"] = allPlayerData.apply(lambda row: categorizeFantasyPoints1(row), axis=1)
    allPlayerData = allPlayerData[['Age','Tm','Pos','H or A','Opp','Last Game','2 Games Ago','3 Games Ago','AvgMP','AvgPTS','AvgREB','AvgAST','AvgSTL','AvgBLK','AvgUSG%','AvgFantScore','Score Category','ScoreCategory1','Fantasy Score']]
    allPlayerData['Age'] = allPlayerData['Age'].str.split("-").str[0]
    # allPlayerData['Date'] = allPlayerData['Date'].str.replace("-","")
    # allPlayerData['Date'] = pd.to_datetime(allPlayerData['Date'],format='%Y%m%d')
    allPlayerData.to_csv("data/FinalData.csv",index=False)

def choosePredictorsAndPrepareData(data):
    data = data[['H or A','Last Game','2 Games Ago','3 Games Ago','AvgMP','AvgUSG%','AvgFantScore','Fantasy Score']]
    data = data.dropna()
    #X_all = data.drop(['Score Category','ScoreCategory1','Fantasy Score'],axis=1)
    X_all = data.drop(['Fantasy Score'],axis=1)
    #Y_all = data['Score Category']
    Y_all = data['Fantasy Score']
    #check which rows have NAN values
    # is_NaN = data.isnull()
    # row_has_NaN = is_NaN.any(axis=1)
    # rows_with_NaN = data[row_has_NaN]
    # print(rows_with_NaN)

    # select predictors
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # sel.fit_transform(X_all)
    # X_all = X_all[X_all.columns[sel.get_support(indices=True)]]

    cols = [['Last Game','2 Games Ago','3 Games Ago','AvgMP','AvgUSG%','AvgFantScore']]
    for col in cols:
        X_all[col] = scale(X_all[col])
    X_all = pd.get_dummies(X_all,columns=["H or A"])
    features = X_all.columns
    if ("H or A_H" not in features):
        X_all['H or A_H'] = 0
    if ("H or A_A" not in features):
        X_all['H or A_A'] = 0
    print(X_all)
    return X_all,Y_all

def testData(data):
    X_all,Y_all = choosePredictorsAndPrepareData(data)
    loaded_model = pickle.load(open('pointsPredictionModel.sav', 'rb'))
    model_features = loaded_model.get_booster().feature_names
    print(model_features)
    return loaded_model.predict(X_all)

def trainData():
    data = pd.read_csv("data/FinalData.csv")

    #drop NAN rows for now
    #data = data[['Tm','Pos','H or A','Opp','Last Game','2 Games Ago','3 Games Ago','AvgMP','AvgUSG%','AvgFantScore','Score Category','ScoreCategory1',"Fantasy Score"]]
    X_all,Y_all = choosePredictorsAndPrepareData(data)

    X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, 
                                                    random_state = 42)
    # Initialize the three models (XGBoost is initialized later)
    clf_A = LogisticRegression(random_state = 42)
    #Boosting refers to this general problem of producing a very accurate prediction rule 
    #by combining rough and moderately inaccurate rules-of-thumb
    clf_B = xgb.XGBClassifier(seed = 82)

    #rf_model = RandomForestClassifier(random_state=5)
    xgb_model = xgb.XGBRegressor(random_state=5)
    xgb_model.fit(X_train,y_train)
    predictions = xgb_model.predict(X_test)
    mae = mean_absolute_error(predictions,y_test)
    print(predictions)
    print(mae)

    filename = 'pointsPredictionModel.sav'
    pickle.dump(xgb_model, open(filename, 'wb'))

    # train_predict(clf_A, X_train, y_train, X_test, y_test)
    # print ('')
    # train_predict(clf_B, X_train, y_train, X_test, y_test)
    # print ('')

    # # TODO: Create the parameters list you wish to tune
    # parameters = { 'learning_rate' : [0.1],
    #             'n_estimators' : [40],
    #             'max_depth': [3],
    #             'min_child_weight': [3],
    #             'gamma':[0.4],
    #             'subsample' : [0.8],
    #             'colsample_bytree' : [0.8],
    #             'scale_pos_weight' : [1],
    #             'reg_alpha':[1e-5]
    #             }  

    # # TODO: Initialize the classifier
    # clf = xgb.XGBClassifier(seed=2)

    # # TODO: Make an f1 scoring function using 'make_scorer' 
    # f1_scorer = make_scorer(f1_score,pos_label='H')

    # # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
    # grid_obj = GridSearchCV(clf,
    #                         scoring=f1_scorer,
    #                         param_grid=parameters,
    #                         cv=5)

    # # TODO: Fit the grid search object to the training data and find the optimal parameters
    # grid_obj = grid_obj.fit(X_train,y_train)

    # # Get the estimator
    # clf = grid_obj.best_estimator_

    # # Report the final F1 score for training and testing after parameter tuning
    # f1, acc = predict_labels(clf, X_train, y_train)
    # print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
        
    # f1, acc = predict_labels(clf, X_test, y_test)
    # print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

def featureSelection():
    data = pd.read_csv("data/FinalData.csv")

    data = data.dropna()

    X_all = data.drop('Score Category',axis=1)
    Y_all = data['Score Category']

    #Center to the mean and component wise scale to unit variance.
    cols = [['Last Game','2 Games Ago','3 Games Ago','AvgMP','Age','AvgPTS','AvgREB','AvgAST','AvgSTL','AvgBLK','AvgORtg','AvgDRtg','AvgTS%','AvgUSG%','AvgeFG%','AvgGmSc%']]
    for col in cols:
        X_all[col] = scale(X_all[col])
    X_all = preprocess_features(X_all)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_stripped = sel.fit_transform(X_all)
    X_all = X_all[X_all.columns[sel.get_support(indices=True)]]
    print(X_all.columns)
    # model = LogisticRegression(random_state=42)
    # rfe = RFE(model)
    # fit = rfe.fit(X_all, Y_all)
    # print("Num Features: %d" % fit.n_features_)
    # print("Selected Features: %s" % fit.support_)
    # print("Feature Ranking: %s" % fit.ranking_)
    #print(X_new)

def main():
    allPlayers = pd.read_csv("data/AllPlayers.csv")
    testing = pd.read_csv("other/testing.csv")
    #allPlayerData = getAllPlayerGameLogs(allPlayers, 1)
    cleanData()
    trainData()
    # featureSelection()

if __name__ == "__main__":
    main()