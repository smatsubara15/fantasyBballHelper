# web scraper found from here: https://m-dendinger19.medium.com/scrape-espn-fantasy-data-with-selenium-4d3b1fdb39f3
# bball reference scraping: https://medium.com/analytics-vidhya/intro-to-scraping-basketball-reference-data-8adcaa79664a
from os import name
import time
from numpy.lib.function_base import select
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import unidecode

import pandas as pd
from urllib.request import urlopen
from datetime import datetime
import re

#todo: put players in for me
def putPlayersIn():
    return

def scrapeESPNFantasyData():
    driver = webdriver.Chrome('/Users/scott/chromedriver')
    driver.get('https://www.espn.com/')
    time.sleep(1)

    ##Open Initial Log In Location
    search_box = driver.find_element_by_id('global-user-trigger')
    search_box.click()
    time.sleep(1)
    print('Open Log In Tab')

    ##Click on the Log In location
    nextbox = driver.find_element_by_xpath("//a[@data-affiliatename='espn']")
    nextbox.click()
    print('Click Login')

    ##Switch to iFrame to enter log in credentials
    time.sleep(1)
    driver.switch_to.frame("disneyid-iframe")
    username = driver.find_element_by_xpath("//input[@placeholder='Username or Email Address']")
    print('Switching to iFrame')

    ##Submit Username and Password
    time.sleep(1)
    username.send_keys('scottmatsubara@gmail.com')
    password = driver.find_element_by_xpath("//input[@placeholder='Password (case sensitive)']")
    password.send_keys('bacon1515')
    time.sleep(1)
    print('Logging In')

    ##Submit credentials
    button = driver.find_element_by_xpath("//button[@class='btn btn-primary btn-submit ng-isolate-scope']")
    button.click()
    driver.page_source

    ##Open Link Page
    time.sleep(1)
    search_box = driver.find_element_by_xpath("//a[@href='/fantasy/']")
    search_box.click()
    print('Going to Fantasy Link')

    ##Selecting Fantasy League
    time.sleep(1)
    leaguego = driver.find_element_by_partial_link_text("MapOTSoul:Yurt7 GhostOfL.S.")
    leaguego.click()
    print('Entering League')

    #Open Players Tab
    time.sleep(1)
    playersgo = driver.find_element_by_xpath("//a[@href='/basketball/players/add?leagueId=1631564033']")
    playersgo.click()
    print("Opening Players Tab")

    site = driver.page_source

    time.sleep(1)
    
    #further Scraping found here: https://realpython.com/beautiful-soup-web-scraper-python/#find-elements-by-html-class-name
    team,position,names,totPts,avgPts = ([] for i in range(5))
    
    for i in range(6):
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')

        # get all the different player info
        team.extend(soup.find_all("span", class_="playerinfo__playerteam"))
        position.extend(soup.find_all("span", class_="playerinfo__playerpos"))
        names.extend(soup.find_all("a", class_="AnchorLink link clr-link pointer"))
        totPts.extend(soup.find_all("div", class_="jsx-2810852873 table--cell total tar sortable"))
        avgPts.extend(soup.find_all("div",class_="jsx-2810852873 table--cell avg tar sortable"))
        time.sleep(1)
        nextPage = driver.find_element_by_xpath("//button[@class='Button Button--default Button--icon-noLabel Pagination__Button Pagination__Button--next']")
        nextPage.click()
        print("Going to Next Page of Players")
        
    driver.quit()
    Players,Teams,Position,totPTS,avgPTS = ([] for i in range(5))
    for i in range(len(names)):
        Players.append(names[i].text)
        Teams.append(team[i].text.upper())
        Position.append(position[i].text)
        totPTS.append(totPts[i].text)
        avgPTS.append(avgPts[i].text)
    
    AvailablePlayers = pd.DataFrame(list(zip(Players,Teams,Position,totPTS,avgPTS)),columns =['Name', 'Team','Position','Total Points','Average Points'])
    AvailablePlayers = AvailablePlayers.drop_duplicates()

    # replace stats with zero if the player didnt play and make the columns numeric
    AvailablePlayers['Total Points'] = AvailablePlayers['Total Points'].replace({"--":'0'})
    AvailablePlayers['Total Points'] = pd.to_numeric(AvailablePlayers['Total Points'])
    AvailablePlayers['Average Points'] = AvailablePlayers['Average Points'].replace({"--":'0'})
    AvailablePlayers['Average Points'] = pd.to_numeric(AvailablePlayers['Average Points'])
    return AvailablePlayers

def scrapeStaticData(url):
    # collect HTML data
    html = urlopen(url)
            
    # create beautiful soup object from HTML
    soup = BeautifulSoup(html, features="lxml")

    # use getText()to extract the headers into a list
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]

    # get rows from table
    rows = soup.findAll('tr')[2:]
    rows_data = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]
    
    # create the dataframe and make sure the number of headers is the same as the number of elements in the row
    statsDf = pd.DataFrame(rows_data, columns = headers[(len(headers)-len(rows_data[0])):]).dropna()
    # print(statsDf)
    return statsDf

def calculateFantasyScore(projectionsDf):
    # created projected average points based on ESPN standard point system
    projectionsDf['Fantasy Score'] = projectionsDf['PTS'] + 2*projectionsDf["FG"] - projectionsDf['FGA'] + projectionsDf['FT'] - projectionsDf['FTA'] \
                                    + projectionsDf['3P'] + projectionsDf['TRB'] + 2 * projectionsDf['AST'] + 4 * projectionsDf['STL'] + 4 * projectionsDf['BLK'] \
                                    - 2 * projectionsDf['TOV']
    return projectionsDf

def scrapeNumberFireData():
    url = "https://www.numberfire.com/nba/fantasy/remaining-projections/?scope=per-game"
    # collect HTML datas
    html = urlopen(url)
            
    # create beautiful soup object from HTML
    soup = BeautifulSoup(html, features="lxml")

    # use getText()to extract the headers into a list
    headers = ['Fantasy Ranking','PTS','G','Min','FG','FGA','FG%','FT','FTA','FT%','3P','3PA','3P%','TRB','AST','STL','BLK','TOV']

    # get rows from table
    rows = soup.findAll('tr')[1:]
    rows_data = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]

    # get all of the data after the empty list
    for i in range(len(rows_data)):
        if rows_data[i]==[]:
            splitVal = i
    playerNames = rows_data[:splitVal]
    playerStats = rows_data[splitVal+1:]

    # get the player name which is within a weird pattern of newlines
    pattern = "\n\n(.*?)\n"
    players = []
    for player in playerNames:
        substring = re.search(pattern, player[0]).group(1)
        players.append(substring)
    
    # create the dataframe and make sure the number of headers is the same as the number of elements in the row
    projectionsDf = pd.DataFrame(playerStats, columns = headers).dropna()

    #make all of the columns numeric
    projectionsDf[headers] = projectionsDf[headers].apply(pd.to_numeric, errors='coerce', axis=1)
    projectionsDf['Name'] = players

    # created projected average points based on ESPN standard point system
    projectionsDf = calculateFantasyScore(projectionsDf)
    projectionsDf = projectionsDf.rename(columns={'Fantasy Score': 'Projected Avg Pts (ROS)'})
    return projectionsDf[['Name','Projected Avg Pts (ROS)']]

def getBballRefHeaders():
    url = "https://www.basketball-reference.com/players/j/jamesle01/gamelog/2021/"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    gamebygame = soup.find('table', id = "pgl_basic")
    gameByGameHeaders = gamebygame.find('thead')

    headers = [th.getText() for th in gameByGameHeaders.find_all('th')]
    headers[5] = "H or A"
    headers[7] = "Result"
    headers = headers[1:]
    headersCSV = pd.DataFrame(headers,columns=["headers"])
    headersCSV.to_csv("Other/bballRefHeaders.csv",index=False)

# return the html data table and the name of the player on the current page
def getGameByGameTable(nameCode,year,advanced):
    if(advanced==True):
        gamelog = "/gamelog-advanced/"
        ID = "pgl_advanced"
    else: 
        gamelog = "/gamelog/"
        ID = "pgl_basic"
    playerUrl = "https://www.basketball-reference.com/players/j/" + nameCode + gamelog + str(year) + "/"
    html = urlopen(playerUrl)
            
    # create beautiful soup object from HTML
    soup = BeautifulSoup(html, features="lxml")

    gamebygame = soup.find('table', id = ID)
    name = soup.find('h1',itemprop = "name")
    name  = name.text.split()
    bballRefName = name[0] + " " + name[1]
    return gamebygame,bballRefName

# get rid of hyphens, periods, accent marks, etc in names
def cleanName(playerName):
    playerNameClean = playerName.replace('.','')
    playerNameClean = playerNameClean.replace('\'','')
    playerNameClean = unidecode.unidecode(playerNameClean)
    return playerNameClean.lower()

# easier to understand data scraping: https://medium.com/analytics-vidhya/how-to-scrape-a-table-from-website-using-python-ce90d0cfb607
# if b2b is true, this will get the players data from the last 3 years
# if it is false, it will get data from the past 10 games
def scrapeBballRefPlayerData(playerName,b2b,advanced):
    # first get rid of periods and apostrophes in names
    playerNameSplit = cleanName(playerName).split()
    playerName = playerNameSplit[0] + " " + playerNameSplit[1]

    # Some names arent in the same pattern as others
    if(playerName =="nicolas claxton"):
        playerName = "nic claxton"
    elif(playerName == "juan hernangomez"):
        playerName = "juancho hernangomez"
    elif(playerName == "cameron thomas"):
        playerName = "cam thomas"
    elif(playerName == "herb jones"):
        playerName = "herbert jones"
    if(playerName == "cedi osman"):
        nameCode = "osmande01"
    elif (playerName == "clint capela"):
        nameCode = "capelca01"
    elif (playerName == "enes freedom"):
        nameCode = "kanteen01"
    elif (playerName == "maxi kleber"):
        nameCode = "klebima01"
    elif (playerName == "didi louzada"):
        nameCode = "louzama01"
    elif (playerName == "frank ntilikina"):
        nameCode = "ntilila01"
    else:
        nameCode = playerNameSplit[1][:5] + playerNameSplit[0][:2] + "01"
    # DNPKeywords = ["Inactive","Did Not Play","Did Not Dress"]
    # DNPList = ['-']*22
    year = 2022
    if(advanced):
        headersDf = pd.read_csv("Other/bballRefHeadersAdvanced.csv")
        headers = list(headersDf["headers"])
    else:
        headersDf = pd.read_csv("Other/bballRefHeaders.csv")
        headers = list(headersDf["headers"])
    #headers = ['G','Date','Age','Tm','H or A','Opp','Result','GS','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','GmSc','+/-']
    playerData = pd.DataFrame(columns=headers)

    #ensure that we ge the right player whether it be names that are similar or sons of players
    nameIncorrect = True
    while(nameIncorrect):
        gamebygame,bballRefName = getGameByGameTable(nameCode,year,advanced)
        # print(playerName)
        # print(bballRefName)
        #check if the page doesnt exist or the players name is not the same as the one on the page
        if ((gamebygame is None) or (cleanName(bballRefName)!=playerName)):
            num = int(nameCode[-1])
            nameCode = nameCode.replace(nameCode[-1],str(num+1))
            nameIncorrect = True
        else:
            nameIncorrect = False
    # if we are finding data for back to back, we get info from the past 2 years, otherwise its just information from the current year
    if b2b:
        numYears = 2
    else: 
        numYears = 1

    for i in range(numYears):
        if(i != 0):
            gamebygame,bballRefName = getGameByGameTable(nameCode,year-i,advanced)

        #checks if the page is empty, meaning that the player didnt play in a season (Usually means the player is a rookie)
        if gamebygame is None:
            return playerData

        data = gamebygame.find('tbody')

        for row in data.find_all('tr'):
            row_data = row.find_all('td')
            gameData = [stat.text for stat in row_data]
            length = len(playerData)
            if(len(gameData)==len(headers)):
                playerData.loc[length] = gameData
            # else:
            #     if(len(gameData)>0):
            #         # if the last element of the row list is one of the DNP keywords, then fill the row list with "-"
            #         if gameData[-1] in DNPKeywords:
            #             gameData = gameData[:-1] + DNPList
            #             playerData.loc[length] = gameData
    #convert all columns that can be converted into ints, ignore errors that say some columns are strings
    playerData = playerData.apply(pd.to_numeric, errors='ignore')
    return playerData
# player projections: http://cs229.stanford.edu/proj2012/Wheeler-PredictingNBAPlayerPerformance.pdf