from time import strptime
from bs4.builder import TreeBuilderRegistry
import numpy as np
import pandas as pd
import csv
from datetime import date,datetime
import datetime as dt
from pandas.io.formats.format import return_docstring
import pytz
from selenium import webdriver
import dataScraper
import predictiveAnalysis
from dataScraper import scrapeESPNFantasyData,scrapeStaticData,scrapeNumberFireData,scrapeBballRefPlayerData
import re
from os.path import exists

abbrFullName = {'ATL':'Atlanta Hawks',
            'BKN':'Brooklyn Nets',
            'BOS':'Boston Celtics',
            'CHA':'Charlotte Hornets',
            'CHI':'Chicago Bulls',
            'CLE':'Cleveland Cavaliers',
            'DAL':'Dallas Mavericks',
            'DEN':'Denver Nuggets',
            'DET':'Detroit Pistons',
            'GS':'Golden State Warriors',
            'HOU':'Houston Rockets',
            'IND':'Indiana Pacers',
            'LAC':'Los Angeles Clippers',
            'LAL':'Los Angeles Lakers',
            'MEM':'Memphis Grizzlies',
            'MIA':'Miami Heat',
            'MIL':'Milwaukee Bucks',
            'MIN':'Minnesota Timberwolves',
            'NO':'New Orleans Pelicans',
            'NY':'New York Knicks',
            'OKC':'Oklahoma City Thunder',
            'ORL':'Orlando Magic',
            'PHI':'Philadelphia 76ers',
            'PHX':'Phoenix Suns',
            'POR':'Portland Trail Blazers',
            'SAC':'Sacramento Kings',
            'SA':'San Antonio Spurs',
            'TOR':'Toronto Raptors',
            'UTAH':'Utah Jazz',
            'WSH':'Washington Wizards'}

fullNameAbbr = {v: k for k, v in abbrFullName.items()}

# get todays date and format it without commas
def getDate():
    # if(isTomorrow):
    #     today = date.today() + dt.timedelta(days=1)
    # else:
    today = date.today()
    today = datetime.now(pytz.timezone("EST"))
    d2 = today.strftime("%B %d, %Y")
    d2 = d2.replace(",","")
    return d2

    #used to help get eastern time and correct format: https://stackoverflow.com/questions/31691007/0-23-hour-military-clock-to-standard-time-hhmm
    # more date stuff: https://www.programiz.com/python-programming/datetime/current-datetime
def getTime():
    tz_NY = pytz.timezone('America/New_York') 
    datetime_NY = datetime.now(tz_NY)
    ET = datetime_NY.strftime("%H:%M:%S")
    ET = datetime.strptime(ET,'%H:%M:%S')
    return ET

#get teams playing on requested days. Returns pd dataframe and boolean stating whether or not it is the next day based on when the earliest game is
def getTeams():
    d2 = getDate()
    #print(d2)
    #format the date so it uses the abbreviated month
    monthAbbr = d2[:3]
    dayYear = d2[-8:]
    todaysDate = monthAbbr + dayYear

    #remove leading 0 if day number is single digit
    if(todaysDate[4]=="0"):
        todaysDate = todaysDate[:4] + todaysDate[5:]

    ET = getTime()
    dayValues = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    dayValuesInput = {"M":0,"T":1,"W":2,"R":3,"F":4,"S":5,"SU":6}
    acceptedDays = ["M","T","W","R","F","S","SU"]
    print("\n Choose which days you want see teams play on: \n M = Monday, T = Tuesday, W = Wednesday, R = Thursday, F = Friday, S = Saturday, Su = Sunday")
    print("Example: T R (This will show you which teams play on both Tuesday and Thursday)")
    print("If you just want to see which teams play on the next back to back days, just hit return")

    # Schedule gotten from: https://www.basketball-reference.com/leagues/NBA_2022_games.html
    # manually exported csv for each month. Can try to automate this.
    schedule =  pd.read_csv("data/NBASchedule.csv")

    # create a new dataframe that we can use to count the number of games per day and to set numerical labels for each day
    gamesPerDay = schedule.groupby('Date',sort=False).size().to_frame('gamesPerDay')
    gamesPerDay['dateCounter'] = np.arange(len(gamesPerDay))

    #combine new dataframe that contains date number with the original dataframe
    schedule = schedule.merge(gamesPerDay,on='Date')

    #split up the date to two separate columns: one with the date and the other with the day of the week
    schedule['ShortenedDate'] = schedule['Date'].str[4:]
    schedule['Day'] = schedule['Date'].str[:3]

    #rename and select columns
    schedule = schedule.rename(columns={'Start (ET)':'Time','Visitor/Neutral':'Visitor','Home/Neutral':'Home'})
    schedule = schedule[["Day","Time","Visitor","Home","ShortenedDate","gamesPerDay","dateCounter"]]
    schedule = schedule.replace({'Visitor':fullNameAbbr,'Home':fullNameAbbr})

    #add leading zero to the Time column
    schedule['Time'] = schedule['Time'].str.rjust(6, "0")
    #Excluding columns if needed: https://www.statology.org/pandas-exclude-column/

    # new dataframe that only includes todays games
    todaysGames = schedule[schedule['ShortenedDate']==todaysDate]

    # record the time of the earliest game and convert it to a time object
    # this will allow us to compare the earliest game time to the current time to see if it is considered
    # the next day yet.
    earliestGame = todaysGames['Time'].iloc[0].upper() + "M"
    earliestGame = datetime.strptime(earliestGame,'%I:%M%p')

    #get todays date code which will be in the first row of the dateCounter column
    dateCodeToday = int(todaysGames['dateCounter'].iloc[0])


    # also get the date code Monday by subtracting todays day value from the dayCode
    # ex: if todays code is 46 and it is Wednesday, Mondays code will be 46-2 or 43
    # this will be used to calculate date codes of all other days of the week
    dateCodeMonday = dateCodeToday - dayValues[todaysGames['Day'].iloc[0]]

    daysChosen = input("Enter Days: ").upper()
    daysChosenList = daysChosen.split()

    # ensure that the inputed days are valid days
    while(not set(daysChosenList).issubset(acceptedDays)):
        print("The days you entered are not valid")
        daysChosen = input("Enter Days: ")
        daysChosenList = daysChosen.split()

    # just in case the user puts the same day twice, make the list of days a set
    daysSet = set(daysChosenList)

    daysToCheck = []
    b2b = False

    # if the user just hits return, the input will be empty so we will search for teams playing in back to back days
    if(len(daysSet)==0):
        b2b = True

        # if the current time is later than the earliest game, the day will be considered the next day
        if(ET.time() > earliestGame.time()):
            # isTomorrow = True
            dateCodeTomorrow = dateCodeToday + 1
        else:
            dateCodeTomorrow = dateCodeToday
        daysToCheck = [dateCodeTomorrow,dateCodeTomorrow+1]

    # otherwise, add days chosen to the list of days to check. 
    else: 
        for day in daysSet:
            dateChosen = dateCodeMonday + dayValuesInput[day]
            if(dateChosen < dateCodeToday):
                dateChosen += 7
            daysToCheck.append(dateChosen)
    # row seletion: https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/

    # back2backDays will consist of all games taking place on the selected days
    back2backDays = schedule[schedule['dateCounter'].isin(daysToCheck)]

    # create a list of all teams playinig in selected days
    Teams = back2backDays['Visitor'].to_list() + back2backDays['Home'].to_list()
    teamsSet = set(Teams)

    teamsPlayingAllDays = []

    # check every team that is playing in the selected days. 
    # If they appear the same amount of times as the number of days selected, then that team plays all all days selected
    for team in teamsSet:
        if(Teams.count(team)==len(daysToCheck)):
            teamsPlayingAllDays.append(team)

    if(b2b == True):  
        print("\n Teams Playing Back to Back:")
    else: 
        print("\n Teams Playing on Your Selected Days:")

    for teams in teamsPlayingAllDays:
        print(teams)
    #teamsAndOpponents = [{k: [] for k in teamsPlayingAllDays}]
    teamsAndOpponents = [teamsPlayingAllDays]
    headers = ["Team"]
    oppCounter = 1
    dateCounters = set(back2backDays["dateCounter"])
    for day in dateCounters:
        headers.append("Opp"+str(oppCounter))
        oppCounter+=1
        opponents = []
        for team in teamsPlayingAllDays:
            matchup = back2backDays[(back2backDays['Visitor']==team)&(back2backDays['dateCounter']==day)]
            if(matchup.empty):
                matchup = back2backDays[(back2backDays['Home']==team)&(back2backDays['dateCounter']==day)]
                opponents.append(list(matchup['Visitor'])[0])
            else:
                opponents.append("@"+list(matchup['Home'])[0])
        teamsAndOpponents.append(opponents)
    
    teamsAndOpponents = pd.DataFrame(teamsAndOpponents)
    teamsAndOpponents = teamsAndOpponents.transpose()
    teamsAndOpponents.columns=headers
    return teamsAndOpponents

# makes sure all names do not include Jr., II, etc. so they all can be compared
def makeNameColumnUniform(data):
    if 'Name' in data.columns:
        data['Name'] = data['Name'].str.split().str[0] + " " + data['Name'].str.split().str[1]
    return data

# split up the first column into 4 appropriate columns
def reorganizeFantasyProsData(fantasyProsData):
    fantasyProsData[['Name','Position']] = fantasyProsData['Player'].str.split('(',1,expand=True)
    fantasyProsData[['Team','Position']] = fantasyProsData['Position'].str.split('-',1,expand=True)
    fantasyProsData = makeNameColumnUniform(fantasyProsData)
    fantasyProsData['Name'] = fantasyProsData['Name'].str.strip()
    fantasyProsData[['Position','Injury Status']] = fantasyProsData['Position'].str.split(')',1,expand=True)
    fantasyProsData['Injury Status'] = fantasyProsData['Injury Status'].replace(r'^\s*$', 'Healthy', regex=True)

    #reoreder the columns:
    col = fantasyProsData.columns.tolist()
    col = col[-4:] + col[1:-4]
    fantasyProsData=fantasyProsData[col]
    return fantasyProsData

# returns average fantasy score on sets of back to back days
def getBack2BackData(playerGamesLog):
    #split the date into year, month, day and make them into int lists
    dateSplit = playerGamesLog['Date'].str.split('-')
    month = list(dateSplit.str[1])
    month = [int(x) for x in month]
    day = list(dateSplit.str[2])
    day = [int(x) for x in day]

    # dict contains the amount of days in every month
    endofMonth = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    back2Back = []

    i=0
    b2bCounter = 1

    #check each date in the log to see if the days are consecutive (which means the player is playing back to back days)
    while i < len(day)-1:
        b2bList = ["b2b"+str(b2bCounter)]*2

        #check if days are numerically consecutive
        if(day[i]+1 == day[i+1]):
            back2Back.extend(b2bList)
            i+=2
            b2bCounter+=1
        
        #check if the first day is the end of the month and the next day is the start of a month
        elif (day[i]==endofMonth[month[i]] and day[i+1]==1):
            back2Back.extend(b2bList)
            i+=2
            b2bCounter+=1

        #append b2b# to the both the rows of the back to back days and append "No" to days without it 
        else:
            back2Back.append("No")
            i+=1

    #append one last "No" if the lengths of days and the back2back list are not the same
    if(len(back2Back)!= len(day)):
        back2Back.append("No")

    playerGamesLog["b2b"] = back2Back

    # Create a dataframe that only consists of back to back games
    b2bGameLog = playerGamesLog[playerGamesLog["b2b"].str.contains("b2b")]
    b2bGameLog = b2bGameLog.apply(pd.to_numeric, errors='ignore')

    b2bGameLog = b2bGameLog.drop(["G","GS"],axis=1)

    # first get the averages of each set of back to back games, then get the average of all back to back games
    b2bGameLog = b2bGameLog.groupby("b2b",sort=False).mean()
    b2bAvg = b2bGameLog.mean()
    b2bAvg = dataScraper.calculateFantasyScore(b2bAvg)
    return(round(b2bAvg['Fantasy Score'],1))

def main():
    bballRefUrl = "https://www.basketball-reference.com/leagues/NBA_2022_per_game.html"
    fantasyProsUrl = "https://www.fantasypros.com/nba/stats/avg-overall.php?days=7"
    date = getDate().replace(' ','') 
    dataScraper.getBballRefHeaders()
    # create specific names for the outputed csv files
    filenameESPN = "data/" + date + "ESPN" + ".csv"
    filenameStats = "data/" + date + "Stats" + ".csv"
    filenameFantasyPros = "data/" + date + "FantasyPros" + ".csv"
    filenameProjections = "data/" + date + "Projections" + ".csv"


    updatedData = input("Do you want updated statistics and available players? ").upper()

    # only scrape the data if the user wants updated statistics
    if(updatedData == 'Y'):
        #espn data gives us which players are available, total and average fantasy points
        availablePlayers = scrapeESPNFantasyData()
        availablePlayers = makeNameColumnUniform(availablePlayers)
        availablePlayers.to_csv(filenameESPN, index=False)

        # used to scrape every players average points if necesary
        # playerStats = scrapeStaticData(bballRefUrl)
        # playerStats.to_csv(filenameStats,index=False)

        # used to get players injury status
        fantasyProsData = scrapeStaticData(fantasyProsUrl)
        fantasyProsData = reorganizeFantasyProsData(fantasyProsData)
        fantasyProsData.to_csv(filenameFantasyPros,index=False)

        # playerProjections = scrapeNumberFireData()
        # playerProjections.to_csv(filenameProjections,index=False)
        playerProjections = pd.read_csv("data/December202021Projections.csv")

    # otherwise just read from the existing csv files
    else:
        availablePlayers = pd.read_csv(filenameESPN)
        # playerStats = pd.read_csv(filenameStats)
        fantasyProsData = pd.read_csv(filenameFantasyPros)
        #playerProjections = pd.read_csv(filenameProjections)
        playerProjections = pd.read_csv("data/December202021Projections.csv")

    # create dataframe that just has information about each players injury status
    injuryStatus = fantasyProsData[['Name','Injury Status']]

    #create the list of players playing on given days
    teamsAndOpponents = getTeams()
    teamsPlaying = list(teamsAndOpponents["Team"])
    playersPlaying=availablePlayers[availablePlayers['Team'].isin(teamsPlaying)]
    if(len(playersPlaying)==0):
        print("None")
        return

    # create full dataframe with averages, injury status, and projections
    playersPlaying = playersPlaying.merge(playerProjections,how='left')
    playersPlaying = playersPlaying.merge(injuryStatus,how='left')
    playersPlaying=playersPlaying.sort_values(by='Total Points',ascending=False)

    # only show 10 healthy/DTD players
    playersShown = 15
    playersHealthy = playersPlaying[playersPlaying["Injury Status"]=="Healthy"]
    playersOut = playersPlaying[playersPlaying["Injury Status"]=="OUT"]
    bestOptions = playersHealthy.head(playersShown)

    #merge the df with best player options with the dataframe showing the teams opponents
    bestOptions = bestOptions.merge(teamsAndOpponents, on = "Team",sort = False)
    bestOptions = bestOptions.sort_values(by="Total Points",ascending=False)

    #change the order of the columns
    headers = bestOptions.columns.tolist()
    headers = headers[:2] + headers[-2:] + headers[2:7]
    bestOptions = bestOptions[headers]

    # get player game logs for every game this season for every player shown and find their B2B Averages
    b2bAverages = []
    for player in bestOptions['Name']:
        b2bPlayerData = scrapeBballRefPlayerData(player,True,False)
        b2bAverages.append(getBack2BackData(b2bPlayerData))
    
    bestOptions.B2BAvg = 0
    bestOptions.B2BAvg = b2bAverages

    # make predictions
    if(input("Do you want to see next day player predictions? Y/N:  ").upper()=="Y"):
        predictiveData = predictiveAnalysis.getAllPlayerGameLogs(bestOptions,0)
        fantasyPointsPredictions = predictiveAnalysis.testData(predictiveData)
        bestOptions["Predicton"] = [round(num) for num in fantasyPointsPredictions]

    # Add column that allows user to search player by No.
    bestOptions.insert(0, 'No.', np.arange(len(bestOptions)))

    # continue this loop until user is finished looking at players
    continueLooking = "Y"
    while(continueLooking == "Y"):
        print("\n")

        # print the best available players and ask if user wants info on specific player
        print(bestOptions.to_string(index=False))
        playerNum = input("\n Enter Player No. to get more information about a specific player. Or press return to exit: ")
        
        # continue if input is a number
        if(playerNum.isnumeric()):
            playerNum = int(playerNum)
            continueLooking = "Y"
        else:
            print("\n Good Luck on Your Matchup This Week!")
            return
        
        # make sure player No. is valid
        while (not (0 <= playerNum <= playersShown)):
            playerNum = int(input("Please enter a valid player number: "))
        player = list(bestOptions['Name'])[playerNum]
        print("\n")

        #scrape player info from the last 10 games and display it
        print(player + " Last 10 Games")
        specificPlayerData = scrapeBballRefPlayerData(player,False,False).tail(10)
        specificPlayerData = dataScraper.calculateFantasyScore(specificPlayerData)
        specificPlayerData = specificPlayerData[['Date','Opp','MP','TRB','AST','STL','BLK','PTS','Fantasy Score']]
        print(specificPlayerData)
        continueLooking = input("Would you like to continue looking at players? Y/N: ").upper()
    print("\n Good Luck on Your Matchup This Week!")
    return 

def pain():
    dataScraper.getBballRefHeaders()
    testing = scrapeBballRefPlayerData("scottie barnes",True,False)
    print(testing)
    # testing['AvgMP'] = predictiveAnalysis.avgMPOverXGames(list(testing["MP"]),3)
    # testing = testing.apply(pd.to_numeric, errors='ignore')
    # testing = dataScraper.calculateFantasyScore(testing)
    # testing["Avg Pts"] = predictiveAnalysis.getAvgSoFar(testing["PTS"])
    # print(testing["Avg Pts"])
    # fantasyProsUrl = "https://www.fantasypros.com/nba/stats/avg-overall.php?days=7"
    # fantasyProsData = scrapeStaticData(fantasyProsUrl)
    # fantasyProsData = reorganizeFantasyProsData(fantasyProsData)

if __name__ == "__main__":
    main()
