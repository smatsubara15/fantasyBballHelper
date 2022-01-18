ESPN FANTASY BASKETBALL HELPER

The simple goal of this project is to allow fantasy managers a simple way to see which NBA players are playing on back to back days. This allows players to maximize the amount of players they have playing in a given week while reducing the amount of pickups they use. This program also allows players to see which players are playing on specifc days that they input. They are given a list containing the players total points, average points, projected ROS Avg Pts, and Injury Status. The user can then input a player number to see stats from a specific players past 10 games. 

TO RUN:
1. Need all necesary libraries: selenium, numpy, pandas, etc
2. Download chromedriver 
3. Go into the dataScraper.py file and Change line 27 to file path to chromedriver
4. Change line 50 and 52 to login info
5. Change line 69 to your team name

PROGRAM: 
- If you would like to get the updated csv of players, enter Y to the first question. The program will use selenium to access and click through Google Chrome. Then it will scrape data for all available players.
- The program will then ask for you to insert the days that you want to find out which teams play on
<img width="814" alt="Screen Shot 2022-01-15 at 12 01 54 AM" src="https://user-images.githubusercontent.com/38053463/149619042-1ce13443-9d4b-4044-8c4e-d8cec09fa014.png">

- The program then produces the following table:
<img width="1147" alt="Screen Shot 2022-01-15 at 12 02 45 AM" src="https://user-images.githubusercontent.com/38053463/149619065-508fe238-9f8a-4d1b-830e-aeab46208c76.png">

- You can look at the past 10 games of a specific player by entering their No. (in this example it is N.)
<img width="595" alt="Screen Shot 2022-01-15 at 12 03 06 AM" src="https://user-images.githubusercontent.com/38053463/149619069-03f1b666-fb6b-4e40-b502-450974436279.png">
