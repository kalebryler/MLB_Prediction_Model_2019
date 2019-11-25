import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import timedelta
import sys
import math
import csv
import pandas as pd
from statistics import mean
import unicodedata
import re
import json
import ast
from pandas.io.json import json_normalize
import numpy as np
from numpy import inf
import warnings
from unidecode import unidecode
import pytz
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
	os.chdir('/Users/kalebryler/Desktop/MLB_Project')
except:
	os.mkdir('/Users/kalebryler/Desktop/MLB_Project')
	os.chdir('/Users/kalebryler/Desktop/MLB_Project')

def merge(d1, d2):
    for k, v in d1.items():
        if k in d2:
            d2[k] = merge(v, d2[k])
    d1.update(d2)
    return d1

def pct_diff(orig, new):
	if orig == new:
	    return 0
	elif orig != 0:
	    return float((abs(new - orig) / orig) * 100.0)
	else:
	    return float((abs(new - 1) / 1) * 100.0)

def avg(li):
	new = [x for x in li if x is not '-']
	new2 = [x for x in new if x is not None]
	new3 = [x for x in new2 if str(x)!='nan']
	new4 = [x for x in new3 if x==x]
	return mean(new3) if len(new4)>0 else 0

def p_diff(orig, new):
	if orig == new:
		return 0
	elif orig != 0:
		a = float(((orig-new)/orig)*100.0)
	else:
		a = -float(((new-1)/1)*100.0)
	return np.log(1+a) if a>0 else -np.log(1+abs(a))

def sum_(li):
	new = [x for x in li if x is not '-']
	new2 = [x for x in new if x is not None]
	new3 = [x for x in new2 if str(x)!='nan']
	new4 = [x for x in new3 if x==x]
	return sum(new3) if len(new4)>0 else 0

def count_(li):
	new = [x for x in li if x is not '-']
	new2 = [x for x in new if x is not None]
	new3 = [x for x in new2 if str(x)!='nan']
	new4 = [x for x in new3 if x==x]
	return len(new3) if len(new4)>0 else 0

def round_up(x, a):
	return np.ceil(x/a)*a

def round_down(x, a):
	return np.floor(x/a)*a

def get_lines(today_date):
	out = {}
	line_url = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb%20odds%202019.xlsx"
	line_data = requests.get(line_url)
	output = open('line_data.xls', 'wb')
	output.write(line_data.content)
	output.close()
	lines = pd.read_excel('line_data.xls')
	lines['Month'] = lines['Date'].astype(str).str[:-2]
	lines['Day'] = lines['Date'].astype(str).str[-2:]
	lines['Date'] = np.where(lines['Month'].astype(int)<10,'0'+lines['Month']+'/'+lines['Day'],lines['Month']+'/'+lines['Day'])
	lines['RL_ML'] = np.select([lines['Unnamed: 18']>0,lines['Unnamed: 18']<0],[lines['Unnamed: 18']/100,-100/lines['Unnamed: 18']],1)
	lines['Over_ML_1'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['Unnamed: 22'],lines['Unnamed: 22'].shift(),lines['Unnamed: 22'].shift(),lines['Unnamed: 22']],0)
	lines['Under_ML_1'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['Unnamed: 22'].shift(-1),lines['Unnamed: 22'],lines['Unnamed: 22'],lines['Unnamed: 22'].shift(-1)],0)
	lines['Over_ML'] = np.select([lines['Over_ML_1']>0,lines['Over_ML_1']<0],[lines['Over_ML_1']/100,-100/lines['Over_ML_1']],1)
	lines['Under_ML'] = np.select([lines['Under_ML_1']>0,lines['Under_ML_1']<0],[lines['Under_ML_1']/100,-100/lines['Under_ML_1']],1)
	lines['ML'] = np.select([lines['Close']>0,lines['Close']<0],[lines['Close']/100,-100/lines['Close']],1)
	lines['OU'] = lines['Close OU']
	lines['RL'] = lines['Run Line']
	lines['F5_OU'] = round_up(lines['OU']/2,0.5)
	lines['Pitcher'],lines['P_Hand'] = lines['Pitcher'].str.split('-',1).str
	lines = lines.replace(['CUB','KAN','SDG','SFO','TAM','WAS','HOW'],['CHC','KC','SD','SF','TB','WSH','HOU'])
	lines['Opp_Team'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['Team'].shift(-1),lines['Team'].shift(),lines['Team'].shift(),lines['Team'].shift(-1)],0)
	lines['Opp_Pitcher'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['Pitcher'].shift(-1),lines['Pitcher'].shift(),lines['Pitcher'].shift(),lines['Pitcher'].shift(-1)],0)
	lines['Opp_P_Hand'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['P_Hand'].shift(-1),lines['P_Hand'].shift(),lines['P_Hand'].shift(),lines['P_Hand'].shift(-1)],0)
	lines['vs_Left'] = np.where(lines['Opp_P_Hand']=='L',1,0)
	lines['vs_Right'] = np.where(lines['Opp_P_Hand']=='R',1,0)
	lines['Opp_1st'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['1st'].shift(-1),lines['1st'].shift(),lines['1st'].shift(),lines['1st'].shift(-1)],0)
	lines['Opp_2nd'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['2nd'].shift(-1),lines['2nd'].shift(),lines['2nd'].shift(),lines['2nd'].shift(-1)],0)
	lines['Opp_3rd'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['3rd'].shift(-1),lines['3rd'].shift(),lines['3rd'].shift(),lines['3rd'].shift(-1)],0)
	lines['Opp_4th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['4th'].shift(-1),lines['4th'].shift(),lines['4th'].shift(),lines['4th'].shift(-1)],0)
	lines['Opp_5th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['5th'].shift(-1),lines['5th'].shift(),lines['5th'].shift(),lines['5th'].shift(-1)],0)
	lines['Opp_6th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['6th'].shift(-1),lines['6th'].shift(),lines['6th'].shift(),lines['6th'].shift(-1)],0)
	lines['Opp_7th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['7th'].shift(-1),lines['7th'].shift(),lines['7th'].shift(),lines['7th'].shift(-1)],0)
	lines['Opp_8th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['8th'].shift(-1),lines['8th'].shift(),lines['8th'].shift(),lines['8th'].shift(-1)],0)
	lines['Opp_9th'] = np.select([lines['VH']=='V',lines['VH']=='H',(lines['VH']=='N')&(lines['Date']==lines['Date'].shift()),(lines['VH']=='N')&(lines['Date']==lines['Date'].shift(-1))],[lines['9th'].shift(-1),lines['9th'].shift(),lines['9th'].shift(),lines['9th'].shift(-1)],0)
	lines = lines[['Date','Team','Pitcher','P_Hand','Opp_Pitcher','vs_Left','vs_Right','ML','RL','RL_ML','OU','F5_OU','Over_ML','Under_ML','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','Opp_1st','Opp_2nd','Opp_3rd','Opp_4th','Opp_5th','Opp_6th','Opp_7th','Opp_8th','Opp_9th']]
	out['Last_Date'] = lines['Date'].iloc[-1]
	out['Current_Date'] = today_date
	lines = lines.set_index(['Date','Team'])
	out['Lines'] = lines
	return out

def get_missing_lines(missing_dates):
	info = {}
	for date in missing_dates:
		try:
			new_date = datetime.strptime(date,"%Y%m%d").strftime("%m/%d")
			url_ml = 'https://www.sportsbookreview.com/betting-odds/mlb-baseball/money-line/?date=' + date
			webaddress_ml = urlopen(url_ml)
			soup_ml = BeautifulSoup(webaddress_ml, 'html.parser')
			ml_data = soup_ml.select("div[class*=_3A-gC]")
			for i in ml_data:
				line = i.text.split('|')[1][3:]
				ml_line  = line.split('%',2)[2] if '%' in line else line.split('--')[1]
				away_team = line.split(' - ')[0]
				away_tuple = tuple([new_date,away_team])
				info[away_tuple] = {}
				away_pitcher_1 = unidecode(line.split(' - ')[1].split(' (')[0].replace('. ','').upper())
				info[away_tuple]['Pitcher'] = away_pitcher_1 if not 'UNDECIDED' in away_pitcher_1 else 'UNDECIDED'
				info[away_tuple]['P_Hand'] = line.split(' - ')[1].split('(')[1].split(')')[0] if info[away_tuple]['Pitcher']!='UNDECIDED' else 'R'
				home_team = line.split(')')[1][3:].split(' - ')[0]
				home_tuple = tuple([new_date,home_team])
				info[home_tuple] = {}
				home_pitcher_1 = unidecode(line.split(home_team)[1].split(' - ')[1].split(' (')[0].replace('. ','').upper())
				info[home_tuple]['Pitcher'] = home_pitcher_1 if not 'UNDECIDED' in home_pitcher_1 else 'UNDECIDED'
				info[home_tuple]['P_Hand'] = line.split(home_team)[1].split('(')[1].split(')')[0] if info[home_tuple]['Pitcher']!='UNDECIDED' else 'R'
				info[away_tuple]['Opp_Pitcher'] = info[home_tuple]['Pitcher']
				info[away_tuple]['vs_Right'] = 1 if info[home_tuple]['P_Hand']=='R' else 0
				info[away_tuple]['vs_Left'] = 1 if info[home_tuple]['P_Hand']=='L' else 0
				info[home_tuple]['Opp_Pitcher'] = info[away_tuple]['Pitcher']
				info[home_tuple]['vs_Right'] = 1 if info[away_tuple]['P_Hand']=='R' else 0
				info[home_tuple]['vs_Left'] = 1 if info[away_tuple]['P_Hand']=='L' else 0
				away_ml_1 = ml_line[:4]
				info[away_tuple]['ML'] = int(away_ml_1[1:])/100 if '+' in away_ml_1 else 100/int(away_ml_1[1:])
				home_ml_1 = ml_line[4:]
				info[home_tuple]['ML'] = int(home_ml_1[1:])/100 if '+' in home_ml_1 else 100/int(home_ml_1[1:])

			url_ou = 'https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/?date=' + date
			webaddress_ou = urlopen(url_ou)
			soup_ou = BeautifulSoup(webaddress_ou, 'html.parser')
			ou_data = soup_ou.select("div[class*=_3A-gC]")
			for i in ou_data:
				line = i.text.split('|')[1][3:]
				away_team = line.split(' - ')[0]
				away_tuple = tuple([new_date,away_team])
				home_team = line.split(')')[1][3:].split(' - ')[0]
				home_tuple = tuple([new_date,home_team])
				ou_line = line.split('%',2)[2].replace('½','.5') if '%' in line else line.split('--')[1].replace('½','.5')
				ou = ou_line.replace('-','A').replace('+','A').split('A')[0]
				info[away_tuple]['OU'] = float(ou)
				info[home_tuple]['OU'] = float(ou)
				over_ml_1 = ou_line.split(ou,1)[1][:4]
				under_ml_1 = ou_line[-4:]
				over_ml = int(over_ml_1[1:])/100 if '+' in over_ml_1 else 100/int(over_ml_1[1:])
				under_ml = int(under_ml_1[1:])/100 if '+' in under_ml_1 else 100/int(under_ml_1[1:])
				info[away_tuple]['Over_ML'] = over_ml
				info[home_tuple]['Over_ML'] = over_ml
				info[away_tuple]['Under_ML'] = under_ml
				info[home_tuple]['Under_ML'] = under_ml

			url_rl = 'https://www.sportsbookreview.com/betting-odds/mlb-baseball/pointspread/?date=' + date
			webaddress_rl = urlopen(url_rl)
			soup_rl = BeautifulSoup(webaddress_rl, 'html.parser')
			rl_data = soup_rl.select("div[class*=_3A-gC]")
			for i in rl_data:
				line = i.text.split('|')[1][3:]
				away_team = line.split(' - ')[0]
				away_tuple = tuple([new_date,away_team])
				home_team = line.split(')')[1][3:].split(' - ')[0]
				home_tuple = tuple([new_date,home_team])
				rl_line = line.split('%',2)[2].replace('½','.5') if '%' in line else line.split('--')[1].replace('½','.5')
				away_rl = rl_line[:4]
				away_rl_ml_1 = rl_line[4:8]
				home_rl = rl_line[8:12]
				home_rl_ml_1 = rl_line[12:16]
				info[away_tuple]['RL'] = float(away_rl) if '-' in away_rl else float(away_rl[1:])
				info[home_tuple]['RL'] = float(home_rl) if '-' in home_rl else float(home_rl[1:])
				away_rl_ml = int(away_rl_ml_1[1:])/100 if '+' in away_rl_ml_1 else 100/int(away_rl_ml_1[1:])
				home_rl_ml = int(home_rl_ml_1[1:])/100 if '+' in home_rl_ml_1 else 100/int(home_rl_ml_1[1:])
				info[away_tuple]['RL_ML'] = away_rl_ml
				info[home_tuple]['RL_ML'] = home_rl_ml
		except:
			pass

	###functionality for scraping inning scores needs to be added for next season###

	lines = pd.DataFrame.from_dict(info,orient='index')

	return lines

def get_game_logs_primary(old_lines,missing_lines,given_date):
	out = {}

	out['Current_Date'] = given_date
	tomorrow = datetime.now(pytz.timezone("America/New_York"))+timedelta(days=1)
	new_tomorrow = tomorrow.strftime("%m/%d")

	lines = pd.concat([old_lines,missing_lines],axis=0,sort=False)
	lines = lines.replace([np.inf, -np.inf], np.nan)

	record_by_date = []
	team_map_1 = {'STL':'St. Louis Cardinals','TOR':'Toronto Blue Jays','LAA':'Los Angeles Angels','NYY':'New York Yankees','ARI':'Arizona Diamondbacks','SD':'San Diego Padres','ATL':'Atlanta Braves','OAK':'Oakland Athletics','BOS':'Boston Red Sox','CLE':'Cleveland Indians','MIA':'Miami Marlins','COL':'Colorado Rockies','MIL':'Milwaukee Brewers','HOU':'Houston Astros','MIN':'Minnesota Twins','CIN':'Cincinnati Reds','NYM':'New York Mets','DET':'Detroit Tigers','PHI':'Philadelphia Phillies','CHC':'Chicago Cubs','SEA':'Seattle Mariners','LAD':'Los Angeles Dodgers','SF':'San Francisco Giants','PIT':'Pittsburgh Pirates','TEX':'Texas Rangers','CWS':'Chicago White Sox','TB':'Tampa Bay Rays','KC':'Kansas City Royals','BAL':'Baltimore Orioles','WSH':'Washington Nationals'}
	team_map_2 = dict((a,b) for b,a in team_map_1.items())
	divisions = {'LAD':'NL West','COL':'NL West','SF':'NL West','ARI':'NL West','SD':'NL West','STL':'NL Central','PIT':'NL Central','CHC':'NL Central','MIL':'NL Central','CIN':'NL Central','WSH':'NL East','MIA':'NL East','NYM':'NL East','PHI':'NL East','ATL':'NL East','LAA':'AL West','OAK':'AL West','HOU':'AL West','TEX':'AL West','SEA':'AL West','MIN':'AL Central','CLE':'AL Central','CWS':'AL Central','KC':'AL Central','DET':'AL Central','NYY':'AL East','BOS':'AL East','TOR':'AL East','TB':'AL East','BAL':'AL East'}
	teams = ['St. Louis Cardinals', 'Toronto Blue Jays', 'Los Angeles Angels', 'New York Yankees', 'Arizona Diamondbacks', 'San Diego Padres', 'Atlanta Braves', 'Oakland Athletics', 'Boston Red Sox', 'Cleveland Indians', 'Miami Marlins', 'Colorado Rockies', 'Milwaukee Brewers', 'Houston Astros', 'Minnesota Twins', 'Cincinnati Reds', 'New York Mets', 'Detroit Tigers', 'Philadelphia Phillies', 'Chicago Cubs', 'Seattle Mariners', 'Los Angeles Dodgers', 'San Francisco Giants', 'Pittsburgh Pirates', 'Texas Rangers', 'Chicago White Sox', 'Tampa Bay Rays', 'Kansas City Royals', 'Baltimore Orioles', 'Washington Nationals']
	
	for i in teams:
		div = divisions[team_map_2[i]]
		new_i = team_map_2[i]

		team_lines = lines[lines.index.get_level_values('Team')==new_i]
		team_lines.index = team_lines.index.get_level_values('Date')
		team_lines = team_lines.loc[~team_lines.index.duplicated(keep='first')]

		url1 = 'https://www.foxsports.com/mlb/'+str(i).replace('.','').replace(' ','-').lower()+'-team-game-log?season=2019&category=HITTER&seasonType=1'
		gl1 = pd.read_html(url1, header=0, index_col='Date')[0]
		gl1 = gl1.loc[~gl1.index.duplicated(keep='first')]
		gl1['PA'] = gl1['AB']+gl1['BB']
		gl1['1B'] = gl1['H']-gl1['2B']-gl1['3B']-gl1['HR']
		gl1['Score'] = (gl1['BB'].mul(0.75)+gl1['1B'].mul(1.15)+gl1['2B'].mul(1.75)+gl1['3B'].mul(2.25)+gl1['HR'].mul(2.75)+gl1['RBI'].mul(2)-gl1['SO'].mul(0.66))/gl1['PA']
		gl1['Hit_Rating'] = (gl1['Score']-0.5)/0.5
		gl1 = gl1[['Opponent','Result','Hit_Rating']]

		gl_temp = pd.concat([team_lines, gl1], axis=1, sort=True)

		url2 = 'https://www.foxsports.com/mlb/'+str(i).replace('.','').replace(' ','-').lower()+'-team-game-log?season=2019&category=PITCHER&seasonType=1'
		gl2 = pd.read_html(url2, header=0, index_col='Date')[0]
		gl2 = gl2.loc[~gl2.index.duplicated(keep='first')]
		gl2 = gl2[['BFP','R','ER','H','HR','BB','SO']]
		gl2['Score'] = (gl2['BB'].mul(0.85)+gl2['H'].mul(1.25)+gl2['HR'].mul(2.15)+gl2['ER'].mul(1.75)+(gl2['R']-gl2['ER']).mul(1.25)-gl2['SO'].mul(0.66))/gl2['BFP']
		gl2['Pitch_Rating'] = (0.5-gl2['Score'])/0.5
		gl2 = gl2[['Pitch_Rating']]

		gl_new = pd.concat([gl_temp, gl2], axis=1, sort=True)
		gl_new = gl_new.loc[~gl_new.index.duplicated(keep='first')][:new_tomorrow]
		gl_new['Net_Rating'] = gl_new['Hit_Rating'] + gl_new['Pitch_Rating']
		gl_new['Drop'] = np.select([(gl_new['Net_Rating']!=gl_new['Net_Rating'])&(gl_new['Net_Rating'].shift(-1)==gl_new['Net_Rating'].shift(-1)),(gl_new['Net_Rating']!=gl_new['Net_Rating'])&(gl_new['Net_Rating'].shift(-2)==gl_new['Net_Rating'].shift(-2))],[np.nan,np.nan],1)
		gl_new = gl_new.dropna(subset=['Drop'])

		gl_new['Opp_Team'] = gl_new['Opponent'].str.split(' ').str[1]
		gl_new['Opp_Team_Divison'] = gl_new['Opp_Team'].map(divisions)
		gl_new['Home'] = np.where(gl_new['Opponent'].str.contains('vs'),1,0)
		gl_new['Win'] = np.where(gl_new['Result'].str.contains('W'),1,-1)
		gl_new['New_Series'] = np.where(gl_new['Opponent']!=gl_new['Opponent'].shift(),1,0)
		gl_new['Game_1'] = np.where(gl_new['New_Series']==1,1,0)
		gl_new['Game_2'] = np.select([(gl_new['Game_1'].shift()==1)&(gl_new['New_Series']==0)],[1],0)
		gl_new['Game_3'] = np.select([(gl_new['Game_2'].shift()==1)&(gl_new['New_Series']==0)],[1],0)
		gl_new['Game_4'] = np.select([(gl_new['Game_3'].shift()==1)&(gl_new['New_Series']==0)],[1],0)
		gl_new['New_Homestand'] = np.select([(gl_new['New_Series']==1)&(gl_new['Home']==1)&(gl_new['Home'].shift()!=1)],[1],0)
		gl_new['New_Homestand_Series'] = np.select([gl_new['New_Homestand']==1,(gl_new['New_Homestand'].shift()==1)&(gl_new['Game_2']==1),(gl_new['New_Homestand'].shift(2)==1)&(gl_new['Game_3']==1),(gl_new['New_Homestand'].shift(3)==1)&(gl_new['Game_4']==1)],[1,1,1,1],0)
		gl_new['New_Road_Trip'] = np.select([(gl_new['New_Series']==1)&(gl_new['Home']==0)&(gl_new['Home'].shift()!=0)],[1],0)
		gl_new['New_Road_Trip_Series'] = np.select([gl_new['New_Road_Trip']==1,(gl_new['New_Road_Trip'].shift()==1)&(gl_new['Game_2']==1),(gl_new['New_Road_Trip'].shift(2)==1)&(gl_new['Game_3']==1),(gl_new['New_Road_Trip'].shift(3)==1)&(gl_new['Game_4']==1)],[1,1,1,1],0)
		gl_new['In_Division'] = np.where(gl_new['Opp_Team_Divison']==div,1,0)
		gl_new['Interleague'] = np.select([('NL' in div)&(gl_new['Opp_Team_Divison'].str.contains('AL')),('AL' in div)&(gl_new['Opp_Team_Divison'].str.contains('NL'))],[1,1],0)
		gl_new['Won_1'] = np.select([gl_new['Win'].shift()==1],[1],0)
		gl_new['Won_2'] = np.select([(gl_new['Win'].shift()==1)&(gl_new['Win'].shift(2)==1)],[1],0)
		gl_new['Won_3'] = np.select([(gl_new['Win'].shift()==1)&(gl_new['Win'].shift(2)==1)&(gl_new['Win'].shift(3)==1)],[1],0)
		gl_new['Won_2_of_3'] = np.where(gl_new['Win'].shift()+gl_new['Win'].shift(2)+gl_new['Win'].shift(3)==1,1,0)
		gl_new['Lost_1'] = np.select([gl_new['Win'].shift()==-1],[1],0)
		gl_new['Lost_2'] = np.select([(gl_new['Win'].shift()==-1)&(gl_new['Win'].shift(2)==-1)],[1],0)
		gl_new['Lost_3'] = np.select([(gl_new['Win'].shift()==-1)&(gl_new['Win'].shift(2)==-1)&(gl_new['Win'].shift(3)==-1)],[1],0)
		gl_new['Lost_2_of_3'] = np.where(gl_new['Win'].shift()+gl_new['Win'].shift(2)+gl_new['Win'].shift(3)==-1,1,0)
		gl_new['Win_Pct'] = gl_new['Win'].shift().cumsum()/gl_new['Drop'].shift().cumsum()
		record_by_date.append(gl_new[['Win_Pct']].copy().rename(columns={'Win_Pct':str(new_i)}))

		gl_new['Scores'] = gl_new['Result'].str[2:]
		gl_new['Final'] = gl_new['Scores'].str.split('-').str[0].astype(float)
		gl_new['Opp_Final'] = gl_new['Scores'].str.split('-').str[1].astype(float)
		gl_new['Total_Runs'] = gl_new['Final']+gl_new['Opp_Final']
		gl_new['F5_Team_Runs'] = gl_new['1st']+gl_new['2nd']+gl_new['3rd']+gl_new['4th']+gl_new['5th']
		gl_new['F5_Opp_Team_Runs'] = gl_new['Opp_1st']+gl_new['Opp_2nd']+gl_new['Opp_3rd']+gl_new['Opp_4th']+gl_new['Opp_5th']
		gl_new['F5_Total_Runs'] = gl_new['F5_Team_Runs']+gl_new['F5_Opp_Team_Runs']
		gl_new['Margin'] = gl_new['Final']-gl_new['Opp_Final']
		gl_new['OU_Margin'] = gl_new['Total_Runs']-gl_new['OU']
		gl_new['Over'] = np.where(gl_new['OU_Margin']>0,1,-1)
		gl_new['F5_OU_Margin'] = gl_new['F5_Total_Runs']-gl_new['F5_OU']
		gl_new['F5_Over'] = np.where(gl_new['F5_OU_Margin']>0,1,-1)
		gl_new['Cover_Margin'] = gl_new['Margin']+gl_new['RL']
		gl_new['Cover'] = np.where(gl_new['Cover_Margin']>0,1,-1)
		gl_new['Blowout_Win'] = np.where((gl_new['Win']==1)&(gl_new['Margin']>3.5),1,0)
		gl_new['Blowout_Loss'] = np.where((gl_new['Win']==-1)&(gl_new['Margin']<-3.5),1,0)
		gl_new['ML_Payout'] = np.where(gl_new['Win']==1,gl_new['ML'],-1)
		gl_new['OU_Payout'] = np.where(gl_new['Over']==1,gl_new['Over_ML'],gl_new['Under_ML'])
		gl_new['F5_OU_Payout'] = np.where(gl_new['F5_Over']==1,gl_new['Over_ML'],gl_new['Under_ML'])
		gl_new['RL_Payout'] = np.where(gl_new['Cover']==1,gl_new['RL_ML'],-1)
		gl_new['True_Win'] = np.select([(gl_new['Win']==1)&(gl_new['Blowout_Win']==1),(gl_new['Win']==1)&(gl_new['Margin']>1.5)&(gl_new['Blowout_Win']==0),(gl_new['Win']==-1)&(gl_new['Margin']<-1.5)&(gl_new['Blowout_Loss']==0),(gl_new['Win']==-1)&(gl_new['Blowout_Loss']==1)],[1.5,1,-1,-1.5],0)
		gl_new['True_Over'] = np.select([(gl_new['Over']==1)&(gl_new['OU_Margin']>2.5),(gl_new['Over']==1)&(gl_new['OU_Margin']>1)&(gl_new['OU_Margin']<=2.5),(gl_new['Over']==-1)&(gl_new['OU_Margin']<-1)&(gl_new['OU_Margin']>=-2.5),(gl_new['Over']==-1)&(gl_new['OU_Margin']<-2.5)],[1.5,1,-1,-1.5],0)
		gl_new['True_F5_Over'] = np.select([(gl_new['F5_Over']==1)&(gl_new['F5_OU_Margin']>1.5),(gl_new['F5_Over']==1)&(gl_new['F5_OU_Margin']>0.5)&(gl_new['F5_OU_Margin']<=1.5),(gl_new['F5_Over']==-1)&(gl_new['F5_OU_Margin']<-0.5)&(gl_new['F5_OU_Margin']>=-1.5),(gl_new['F5_Over']==-1)&(gl_new['F5_OU_Margin']<-1.5)],[1.5,1,-1,-1.5],0)
		gl_new['True_Cover'] = np.select([(gl_new['Cover']==1)&(gl_new['Cover_Margin']>2.5),(gl_new['Cover']==1)&(gl_new['Cover_Margin']>1)&(gl_new['Cover_Margin']<=2.5),(gl_new['Cover']==-1)&(gl_new['Cover_Margin']<-1)&(gl_new['Cover_Margin']>=-2.5),(gl_new['Cover']==-1)&(gl_new['Cover_Margin']<-2.5)],[1.5,1,-1,-1.5],0)
		gl_new['True_Hit'] = np.select([gl_new['True_Win']==1.5,gl_new['True_Win']==1,gl_new['True_Win']==-1,gl_new['True_Win']==-1.5],[gl_new['Hit_Rating']+0.2,gl_new['Hit_Rating']+0.1,gl_new['Hit_Rating']-0.1,gl_new['Hit_Rating']-0.2],gl_new['Hit_Rating'])
		gl_new['True_Pitch'] = np.select([gl_new['True_Win']==1.5,gl_new['True_Win']==1,gl_new['True_Win']==-1,gl_new['True_Win']==-1.5],[gl_new['Pitch_Rating']+0.2,gl_new['Pitch_Rating']+0.1,gl_new['Pitch_Rating']-0.1,gl_new['Pitch_Rating']-0.2],gl_new['Pitch_Rating'])
		gl_new.drop(['Drop', 'Scores'], axis=1, inplace=True)
		out[i] = gl_new.to_json(orient='index')

	opp_records = pd.concat(record_by_date,axis=1,sort=False)
	out['Record_By_Date'] = opp_records.to_json(orient='index')

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')

	with open('game_logs_primary.txt', 'w') as file:
		file.write(json.dumps(out))

	return out

def get_game_logs_secondary(primary_logs,given_date):
	out = {}
	out['Current_Date'] = given_date

	team_map_1 = {'STL':'St. Louis Cardinals','TOR':'Toronto Blue Jays','LAA':'Los Angeles Angels','NYY':'New York Yankees','ARI':'Arizona Diamondbacks','SD':'San Diego Padres','ATL':'Atlanta Braves','OAK':'Oakland Athletics','BOS':'Boston Red Sox','CLE':'Cleveland Indians','MIA':'Miami Marlins','COL':'Colorado Rockies','MIL':'Milwaukee Brewers','HOU':'Houston Astros','MIN':'Minnesota Twins','CIN':'Cincinnati Reds','NYM':'New York Mets','DET':'Detroit Tigers','PHI':'Philadelphia Phillies','CHC':'Chicago Cubs','SEA':'Seattle Mariners','LAD':'Los Angeles Dodgers','SF':'San Francisco Giants','PIT':'Pittsburgh Pirates','TEX':'Texas Rangers','CWS':'Chicago White Sox','TB':'Tampa Bay Rays','KC':'Kansas City Royals','BAL':'Baltimore Orioles','WSH':'Washington Nationals'}
	team_map_2 = dict((a,b) for b,a in team_map_1.items())
	teams = ['St. Louis Cardinals', 'Toronto Blue Jays', 'Los Angeles Angels', 'New York Yankees', 'Arizona Diamondbacks', 'San Diego Padres', 'Atlanta Braves', 'Oakland Athletics', 'Boston Red Sox', 'Cleveland Indians', 'Miami Marlins', 'Colorado Rockies', 'Milwaukee Brewers', 'Houston Astros', 'Minnesota Twins', 'Cincinnati Reds', 'New York Mets', 'Detroit Tigers', 'Philadelphia Phillies', 'Chicago Cubs', 'Seattle Mariners', 'Los Angeles Dodgers', 'San Francisco Giants', 'Pittsburgh Pirates', 'Texas Rangers', 'Chicago White Sox', 'Tampa Bay Rays', 'Kansas City Royals', 'Baltimore Orioles', 'Washington Nationals']

	all_teams = []
	record_by_date = pd.read_json(primary_logs['Record_By_Date'],orient='index')

	for i in teams:

		gl = pd.read_json(primary_logs[i],orient='index')
		gl = gl.dropna(subset=['Opp_Team'])
		gl['Series_Win'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_Win'].shift(),gl['True_Win'].shift().rolling(2,min_periods=1).sum(),gl['True_Win'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_Hit'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_Hit'].shift(),gl['True_Hit'].shift().rolling(2,min_periods=1).sum(),gl['True_Hit'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_Pitch'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_Pitch'].shift(),gl['True_Pitch'].shift().rolling(2,min_periods=1).sum(),gl['True_Pitch'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_Net'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['Net_Rating'].shift(),gl['Net_Rating'].shift().rolling(2,min_periods=1).sum(),gl['Net_Rating'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_Over'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_Over'].shift(),gl['True_Over'].shift().rolling(2,min_periods=1).sum(),gl['True_Over'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_F5_Over'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_F5_Over'].shift(),gl['True_F5_Over'].shift().rolling(2,min_periods=1).sum(),gl['True_F5_Over'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Series_Cover'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[0,gl['True_Cover'].shift(),gl['True_Cover'].shift().rolling(2,min_periods=1).sum(),gl['True_Cover'].shift().rolling(3,min_periods=1).sum()],0)
		gl['Opp_Win_Pct'] = record_by_date.lookup(gl.index.astype(str), gl['Opp_Team'])
		gl['Win_Pct_Diff'] = gl['Win_Pct']-gl['Opp_Win_Pct']
		gl['Team_Good'] = np.where(gl['Win_Pct']>0.1,1,0)
		gl['Team_Middle'] = np.where(abs(gl['Win_Pct'])<0.1,1,0)
		gl['Team_Bad'] = np.where(gl['Win_Pct']<-0.1,1,0)
		gl['Opp_Team_Good'] = np.where(gl['Opp_Win_Pct']>0.1,1,0)
		gl['Opp_Team_Middle'] = np.where(abs(gl['Opp_Win_Pct'])<0.1,1,0)
		gl['Opp_Team_Bad'] = np.where(gl['Opp_Win_Pct']<-0.1,1,0)
		gl['Favorite'] = np.where(gl['ML']<0.8,1,0)
		gl['Close'] = np.where((gl['ML']>=0.8)&(gl['ML']<=1.2),1,0)
		gl['Underdog'] = np.where(gl['ML']>1.2,1,0)
		gl['Matchup_Profile'] = np.select([gl['In_Division']==1,gl['Interleague']==1],['In_Division','Interleague'],'In_League')
		gl['Series_Win_Classifier'] = np.select([gl['Series_Win']>1,gl['Series_Win']>0.33,gl['Series_Win']<-1,gl['Series_Win']<-0.33],[2,1,-1,-2],0)
		gl['Series_Hit_Classifier'] = np.select([gl['Series_Hit']>0.75,gl['Series_Hit']>0.2,gl['Series_Hit']<-0.75,gl['Series_Hit']<-0.2],[2,1,-1,-2],0)
		gl['Series_Pitch_Classifier'] = np.select([gl['Series_Pitch']>0.75,gl['Series_Pitch']>0.2,gl['Series_Pitch']<-0.75,gl['Series_Pitch']<-0.2],[2,1,-1,-2],0)
		gl['Series_Over_Classifier'] = np.select([gl['Series_Over']>1,gl['Series_Over']>0.33,gl['Series_Over']<-1,gl['Series_Over']<-0.33],[2,1,-1,-2],0)
		gl['Series_F5_Over_Classifier'] = np.select([gl['Series_F5_Over']>1,gl['Series_F5_Over']>0.33,gl['Series_F5_Over']<-1,gl['Series_F5_Over']<-0.33],[2,1,-1,-2],0)
		gl['Series_Cover_Classifier'] = np.select([gl['Series_Cover']>1,gl['Series_Cover']>0.33,gl['Series_Cover']<-1,gl['Series_Cover']<-0.33],[2,1,-1,-2],0)
		gl['Game_Number'] = np.select([gl['Game_1']==1,gl['Game_2']==1,gl['Game_3']==1,gl['Game_4']==1],[1,2,3,4],0)
		gl['Betting_Profile'] = np.select([gl['Favorite']==1,gl['Underdog']==1,gl['Close']==1],['Favorite','Underdog','Close'])
		gl['Team_Profile'] = np.select([gl['Team_Good']==1,gl['Team_Bad']==1,gl['Team_Middle']==1],['Good','Bad','Middle'])
		gl['Opp_Team_Profile'] = np.select([gl['Opp_Team_Good']==1,gl['Opp_Team_Bad']==1,gl['Opp_Team_Middle']==1],['Good','Bad','Middle'])
		gl = gl.replace([np.inf, -np.inf], np.nan)
		gl.fillna(0,inplace=True)
		gl.drop(['Team_Good','Team_Middle','Team_Bad','Opp_Team_Good','Opp_Team_Middle','Opp_Team_Bad','Favorite','Close','Underdog'], axis=1, inplace=True)

		for stat in ['Win','Hit','Pitch','Over','F5_Over','Cover']:

			new_stat = 'True_'+str(stat)

			overall_1 = 'Overall_30_'+str(stat)
			gl[overall_1] = gl[new_stat].shift().rolling(30,min_periods=1).mean()

			overall_2 = 'Overall_15_'+str(stat)
			gl[overall_2] = gl[new_stat].shift().rolling(15,min_periods=1).mean()

			home_away = 'Home_Away_'+str(stat)
			gl[home_away] = gl.groupby('Home')[new_stat].transform(lambda x : x.shift().rolling(20,min_periods=1).mean())

			right_left = 'Right_Left_'+str(stat)
			gl[right_left] = gl.groupby('vs_Right')[new_stat].transform(lambda x: x.shift().rolling(15,min_periods=1).mean())

			series_by_game = 'Series_By_Game_'+str(stat)
			gl[series_by_game] = gl.groupby(['Game_Number','Series_Win_Classifier'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(15,min_periods=1).mean())

			betting_profile = 'Betting_Profile_'+str(stat)
			gl[betting_profile] = gl.groupby('Betting_Profile')[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean())

			opp_team_profile = 'Opp_Team_Profile_'+str(stat)
			gl[opp_team_profile] = gl.groupby('Opp_Team_Profile')[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean())

			matchup_profile = 'Matchup_Profile_'+str(stat)
			gl[matchup_profile] = gl.groupby(['Matchup_Profile','Home'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean())

			for num in range(1,6):

				overall = 'Overall_'+str(stat)+'_'+str(num)
				gl[overall] = gl[new_stat].shift(num)

				home_away = 'Home_Away_'+str(stat)+'_'+str(num)
				gl[home_away] = gl.groupby('Home')[new_stat].shift(num)

				right_left = 'Right_Left_'+str(stat)+'_'+str(num)
				gl[right_left] = gl.groupby('vs_Right')[new_stat].shift(num)

				pitcher = 'Pitcher_'+str(stat)+'_'+str(num)
				gl[pitcher] = gl.groupby('Pitcher')[new_stat].shift(num)

				pitcher_home_away = 'Pitcher_Home_Away_'+str(stat)+'_'+str(num)
				gl[pitcher_home_away] = gl.groupby(['Pitcher','Home'],as_index=False)[new_stat].shift(num)

				series_by_game = 'Series_By_Game_'+str(stat)+'_'+str(num)
				gl[series_by_game] = gl.groupby(['Game_Number','Series_Win_Classifier'],as_index=False)[new_stat].shift(num)

				betting_profile = 'Betting_Profile_'+str(stat)+'_'+str(num)
				gl[betting_profile] = gl.groupby('Betting_Profile')[new_stat].shift(num)

				opp_team_profile = 'Opp_Team_Profile_'+str(stat)+'_'+str(num)
				gl[opp_team_profile] = gl.groupby('Opp_Team_Profile')[new_stat].shift(num)

				matchup_profile = 'Matchup_Profile_'+str(stat)+'_'+str(num)
				gl[matchup_profile] = gl.groupby(['Matchup_Profile','Home'],as_index=False)[new_stat].shift(num)

				v_team = 'vs_Team_'+str(stat)+'_'+str(num)
				gl[v_team] = gl.groupby('Opp_Team')[new_stat].shift(num)

				pitcher_betting_profile = 'Pitcher_Betting_Profile_'+str(stat)+'_'+str(num)
				gl[pitcher_betting_profile] = gl.groupby(['Pitcher','Betting_Profile'],as_index=False)[new_stat].shift(num)

				pitcher_opp_team_profile = 'Pitcher_Opp_Team_Profile_'+str(stat)+'_'+str(num)
				gl[pitcher_opp_team_profile] = gl.groupby(['Pitcher','Opp_Team_Profile'],as_index=False)[new_stat].shift(num)

				pitcher_matchup_profile = 'Pitcher_Matchup_Profile_'+str(stat)+'_'+str(num)
				gl[pitcher_matchup_profile] = gl.groupby(['Pitcher','Matchup_Profile','Home'],as_index=False)[new_stat].shift(num)

				pitcher_v_team = 'Pitcher_vs_Team_'+str(stat)+'_'+str(num)
				gl[pitcher_v_team] = gl.groupby(['Pitcher','Opp_Team'],as_index=False)[new_stat].shift(num)

		gl['Team_Name'] = i
		all_teams.append(gl)

	total_df = pd.concat(all_teams,axis=0)
	total_df = total_df.reset_index()
	total_df.rename(columns={'index': 'Date'}, inplace=True)
	total_df.set_index(['Date','Team_Name'],inplace=True)
	total_df = total_df.sort_values('Date')

	for stat in ['Win','Hit','Pitch','Over','F5_Over','Cover']:

		new_stat = 'True_'+str(stat)

		opp_team_overall_1 = 'Opp_Team_Overall_30_'+str(stat)
		total_df[opp_team_overall_1] = total_df.groupby('Opp_Team')[new_stat].transform(lambda x: x.shift().rolling(30,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_overall_2 = 'Opp_Team_Overall_15_'+str(stat)
		total_df[opp_team_overall_2] = total_df.groupby('Opp_Team')[new_stat].transform(lambda x: x.shift().rolling(15,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_home_away = 'Opp_Team_Home_Away_'+str(stat)
		total_df[opp_team_home_away] = total_df.groupby(['Opp_Team','Home'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_right_left = 'Opp_Team_Right_Left_'+str(stat)
		total_df[opp_team_right_left] = total_df.groupby(['Opp_Team','P_Hand'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_series_by_game = 'Opp_Team_Series_By_Game_'+str(stat)
		total_df[opp_team_series_by_game] = total_df.groupby(['Opp_Team','Game_Number','Series_Win_Classifier'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(15,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_betting_profile = 'Opp_Team_Betting_Profile_'+str(stat)
		total_df[opp_team_betting_profile] = total_df.groupby(['Opp_Team','Betting_Profile'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_opp_team_profile = 'Opp_Team_Opp_Team_Profile_'+str(stat)
		total_df[opp_team_opp_team_profile] = total_df.groupby(['Opp_Team','Opp_Team_Profile'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean().reset_index(0,drop=True))

		opp_team_matchup_profile = 'Opp_Team_Matchup_Profile_'+str(stat)
		total_df[opp_team_matchup_profile] = total_df.groupby(['Opp_Team','Matchup_Profile','Home'],as_index=False)[new_stat].transform(lambda x: x.shift().rolling(20,min_periods=1).mean().reset_index(0,drop=True))


		for num in range(1,6):

			opp_team_overall = 'Opp_Team_Overall_'+str(stat)+'_'+str(num)
			total_df[opp_team_overall] = total_df.groupby('Opp_Team')[new_stat].shift(num)

			opp_team_home_away = 'Opp_Team_Home_Away_'+str(stat)+'_'+str(num)
			total_df[opp_team_home_away] = total_df.groupby(['Opp_Team','Home'],as_index=False)[new_stat].shift(num)

			opp_team_right_left = 'Opp_Team_Right_Left_'+str(stat)+'_'+str(num)
			total_df[opp_team_right_left] = total_df.groupby(['Opp_Team','P_Hand'],as_index=False)[new_stat].shift(num)

			opp_pitcher = 'Opp_Pitcher_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher] = total_df.groupby('Opp_Pitcher')[new_stat].shift(num)

			opp_pitcher_home_away = 'Opp_Pitcher_Home_Away_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher_home_away] = total_df.groupby(['Opp_Pitcher','Home'],as_index=False)[new_stat].shift(num)
 
			opp_team_series_by_game = 'Opp_Team_Series_By_Game_'+str(stat)+'_'+str(num)
			total_df[opp_team_series_by_game] = total_df.groupby(['Opp_Team','Game_Number','Series_Win_Classifier'],as_index=False)[new_stat].shift(num)

			opp_team_betting_profile = 'Opp_Team_Betting_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_team_betting_profile] = total_df.groupby(['Opp_Team','Betting_Profile'],as_index=False)[new_stat].shift(num)

			opp_team_opp_team_profile = 'Opp_Team_Opp_Team_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_team_opp_team_profile] = total_df.groupby(['Opp_Team','Opp_Team_Profile'],as_index=False)[new_stat].shift(num)

			opp_team_matchup_profile = 'Opp_Team_Matchup_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_team_matchup_profile] = total_df.groupby(['Opp_Team','Matchup_Profile','Home'],as_index=False)[new_stat].shift(num)

			opp_pitcher_betting_profile = 'Opp_Pitcher_Betting_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher_betting_profile] = total_df.groupby(['Opp_Pitcher','Betting_Profile'],as_index=False)[new_stat].shift(num)

			opp_pitcher_opp_team_profile = 'Opp_Pitcher_Opp_Team_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher_opp_team_profile] = total_df.groupby(['Opp_Pitcher','Team_Profile'],as_index=False)[new_stat].shift(num)

			opp_pitcher_matchup_profile = 'Opp_Pitcher_Matchup_Profile_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher_matchup_profile] = total_df.groupby(['Opp_Pitcher','Matchup_Profile','Home'],as_index=False)[new_stat].shift(num)

			opp_pitcher_v_team = 'Opp_Pitcher_vs_Team_'+str(stat)+'_'+str(num)
			total_df[opp_pitcher_v_team] = total_df.groupby(['Opp_Pitcher','Team_Name'],as_index=False)[new_stat].shift(num)

	for i in teams:
		gl = total_df[total_df.index.get_level_values('Team_Name')==i]
		gl = gl.reset_index()
		gl.set_index('Date')
		out[i] = gl.to_json(orient='index')

	total_df = total_df.reset_index()
	total_df['New_Index'] = total_df['Date'] + ' ' + total_df['Team_Name']
	total_df.set_index('New_Index',inplace=True)

	out['All_Teams'] = total_df.to_json(orient='index')

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')

	with open('game_logs.txt', 'w') as file:
		file.write(json.dumps(out))

	return out

def get_logs(t_delta):
	today_date = datetime.now(pytz.timezone("America/New_York"))-timedelta(days=t_delta)
	if today_date.month > 10 or (today_date.month < 4 and today_date.day < 15):
		new_date = "OFFSEASON"
	else:
		new_date = today_date.strftime("%m/%d")

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')

	try:
		hist_logs = open('game_logs.txt', 'r')
		game_logs = json.load(hist_logs)
		if game_logs['Current_Date'] == new_date:
			print('Game Logs Found')
		else:
			try:
				primary_logs = open('game_logs_primary.txt', 'r')
				primary_game_logs = json.load(primary_logs)
				if primary_game_logs['Current_Date'] == new_date:
					print('Primary Game Logs Found')
					game_logs = get_game_logs_secondary(primary_game_logs,new_date)
					print('Secondary Game Logs Found')
				else:
					lines = get_lines(new_date)
					print('Lines Found')
					if new_date == "OFFSEASON":
						missing_dates = []
					else:
						missing_dates = []
						last = datetime.strptime(lines['Last_Date'],"%m/%d")
						current = datetime.strptime(new_date,"%m/%d")
						while last < current:
						    last += timedelta(days=1)
						    missing_dates.append(last.strftime("2019%m%d"))
					missing_lines = get_missing_lines(missing_dates)
					print('Missing Lines Found')
					primary_game_logs = get_game_logs_primary(lines['Lines'],missing_lines,new_date)
					print('Primary Game Logs Found')
					game_logs = get_game_logs_secondary(primary_game_logs,new_date)
					print('Secondary Game Logs Found')
			except:
				lines = get_lines(new_date)
				print('Lines Found')
				if new_date == "OFFSEASON":
					missing_dates = []
				else:
					missing_dates = []
					last = datetime.strptime(lines['Last_Date'],"%m/%d")
					current = datetime.strptime(new_date,"%m/%d")
					while last < current:
					    last += timedelta(days=1)
					    missing_dates.append(last.strftime("2019%m%d"))
				missing_lines = get_missing_lines(missing_dates)
				print('Missing Lines Found')
				primary_game_logs = get_game_logs_primary(lines['Lines'],missing_lines,new_date)
				print('Primary Game Logs Found')
				game_logs = get_game_logs_secondary(primary_game_logs,new_date)
				print('Secondary Game Logs Found')
	except:
		try:
			primary_logs = open('game_logs_primary.txt', 'r')
			primary_game_logs = json.load(primary_logs)
			if primary_game_logs['Current_Date'] == new_date:
				print('Primary Game Logs Found')
				game_logs = get_game_logs_secondary(primary_game_logs,new_date)
				print('Secondary Game Logs Found')
			else:
				lines = get_lines(new_date)
				print('Lines Found')
				if new_date == "OFFSEASON":
					missing_dates = []
				else:
					missing_dates = []
					last = datetime.strptime(lines['Last_Date'],"%m/%d")
					current = datetime.strptime(new_date,"%m/%d")
					while last < current:
					    last += timedelta(days=1)
					    missing_dates.append(last.strftime("2019%m%d"))
				missing_lines = get_missing_lines(missing_dates)
				print('Missing Lines Found')
				primary_game_logs = get_game_logs_primary(lines['Lines'],missing_lines,new_date)
				print('Primary Game Logs Found')
				game_logs = get_game_logs_secondary(primary_game_logs,new_date)
				print('Secondary Game Logs Found')
		except:
			lines = get_lines(new_date)
			print('Lines Found')
			if new_date == "OFFSEASON":
				missing_dates = []
			else:
				missing_dates = []
				last = datetime.strptime(lines['Last_Date'],"%m/%d")
				current = datetime.strptime(new_date,"%m/%d")
				while last < current:
				    last += timedelta(days=1)
				    missing_dates.append(last.strftime("2019%m%d"))
			missing_lines = get_missing_lines(missing_dates)
			print('Missing Lines Found')
			primary_game_logs = get_game_logs_primary(lines['Lines'],missing_lines,new_date)
			print('Primary Game Logs Found')
			game_logs = get_game_logs_secondary(primary_game_logs,new_date)
			print('Secondary Game Logs Found')
	return game_logs

def get_logs_from_primary(t_delta):
	today_date = datetime.now(pytz.timezone("America/New_York"))-timedelta(days=t_delta)
	if today_date.month > 10 or (today_date.month < 4 and today_date.day < 15):
		new_date = "OFFSEASON"
	else:
		new_date = today_date.strftime("%m/%d")

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/Game_Log_Files')

	primary_logs = open('game_logs_primary.txt', 'r')
	primary_game_logs = json.load(primary_logs)
	print('Primary Game Logs Found')
	game_logs = get_game_logs_secondary(primary_game_logs,new_date)
	print('Secondary Game Logs Found')
	return game_logs

class GameLogs:
	def __init__(self,timedelta=None):
		self.timedelta = timedelta if timedelta != None else 0
		self.log_txt = get_logs_from_primary(self.timedelta)

def write_game_logs():
	game_logs = GameLogs(0)

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')

	teams = ['All_Teams','St. Louis Cardinals', 'Toronto Blue Jays', 'Los Angeles Angels', 'New York Yankees', 'Arizona Diamondbacks', 'San Diego Padres', 'Atlanta Braves', 'Oakland Athletics', 'Boston Red Sox', 'Cleveland Indians', 'Miami Marlins', 'Colorado Rockies', 'Milwaukee Brewers', 'Houston Astros', 'Minnesota Twins', 'Cincinnati Reds', 'New York Mets', 'Detroit Tigers', 'Philadelphia Phillies', 'Chicago Cubs', 'Seattle Mariners', 'Los Angeles Dodgers', 'San Francisco Giants', 'Pittsburgh Pirates', 'Texas Rangers', 'Chicago White Sox', 'Tampa Bay Rays', 'Kansas City Royals', 'Baltimore Orioles', 'Washington Nationals']

	for i in teams:
		team = pd.read_json(game_logs.log_txt[i],orient='index')
		team.set_index('Date',inplace=True)
		file_name = i.replace(' ','_').replace('.','') + ".csv"
		team.to_csv(file_name)

	print("Game Logs Written")

###Write Game Logs###
write_game_logs()
