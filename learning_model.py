import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from copy import deepcopy

def round_up(x, a):
	return int(np.ceil(x/a)*a)

def round_down(x, a):
	return int(np.floor(x/a)*a)

class NeuralNet:
	def __init__(self,x,y,ml,info):
		classifier = ExtraTreesClassifier(n_estimators=int(x.shape[1]/2))
		classifier = classifier.fit(x[:-1], y.ravel()[:-1])
		self.trim = SelectFromModel(classifier, prefit=True)
		x = self.trim.transform(x)

		self.scaler = StandardScaler()
		self.scaler.fit(x[:-1])
		x = self.scaler.transform(x)

		self.encoder = preprocessing.LabelEncoder()
		self.encoder.fit(y.ravel()[:-1])
		y = self.encoder.transform(y.ravel())

		self.x_game = x[-1]
		self.y_game = y[-1]
		self.ml_game = ml[-1]
		self.info_game = info[-1]

		self.x = x[:-1]
		self.y = y[:-1]
		self.ml = ml[:-1]
		self.info = info[:-1]

		self.x_train,self.x_test,self.y_train,self.y_test,self.ml_train,self.ml_test,self.info_train,self.info_test = train_test_split(self.x,self.y,self.ml,self.info,test_size = 0.25)

		self.input_size = self.x.shape[1]
		self.output_size = len(set(self.y_train))

	def model(self):
		net = MLPClassifier(hidden_layer_sizes=(round_up(self.input_size/2,100),round_up(self.input_size/2,100),round_up(self.input_size/2,100)), activation='relu', solver='adam', max_iter=10000)
		self.fit = net.fit(self.x_train, self.y_train.ravel())
		predictions = self.fit.predict(self.x_test)

		if self.ml.shape[1] == 1:
			self.results = pd.DataFrame([predictions,self.y_test,self.ml_test.ravel()]).T
			self.results.columns = ['Predicted','Actual','ML']
			self.results['Error'] = abs(self.results['Predicted']-self.results['Actual'])
			self.results['Success'] = np.where(self.results['Predicted']==self.results['Actual'],1,0)
			self.results['Payout'] = np.select([self.results['Success']==1,self.results['Success']==0],[self.results['ML'],-1],0)

			self.accuracy = self.results['Success'].mean()
			self.pct_return = self.results['Payout'].mean()
			self.profit = self.results['Payout'].sum()

		else:
			self.results = pd.DataFrame([predictions,self.y_test,self.ml_test[:,0],self.ml_test[:,1]]).T
			self.results.columns = ['Predicted','Actual','Over_ML','Under_ML']
			self.results['Error'] = abs(self.results['Predicted']-self.results['Actual'])
			self.results['Success'] = np.where(self.results['Predicted']==self.results['Actual'],1,0)
			self.results['Payout'] = np.select([self.results['Success']==1,self.results['Success']==0],[self.results['Predicted']*self.results['Over_ML']+(1-self.results['Predicted'])*self.results['Under_ML'],-1],0)

			self.accuracy = self.results['Success'].mean()
			self.pct_return = self.results['Payout'].mean()
			self.profit = self.results['Payout'].sum()

	def predict_game(self):
		predictions = self.fit.predict([self.x_game])

		if self.ml.shape[1] == 1:
			self.game_results = pd.DataFrame([self.info_game.ravel(),predictions,self.y_game.ravel(),self.ml_game.ravel(),self.accuracy.ravel()]).T
			self.game_results.columns = ['Matchup','Predicted','Actual','ML','Test_Accuracy']
			self.game_results['Success'] = np.where(self.game_results['Predicted']==self.game_results['Actual'],1,0)
			self.game_results['Payout'] = np.select([self.game_results['Success']==1,self.game_results['Success']==0],[self.game_results['ML'],-1],0)

		else:
			self.game_results = pd.DataFrame([self.info_game.ravel(),predictions,self.y_game.ravel(),self.ml_game[0].ravel(),self.ml_game[1].ravel(),self.accuracy.ravel()]).T
			self.game_results.columns = ['Matchup','Predicted','Actual','Over_ML','Under_ML','Test_Accuracy']
			self.game_results['Success'] = np.where(self.game_results['Predicted']==self.game_results['Actual'],1,0)
			self.game_results['Payout'] = np.select([self.game_results['Success']==1,self.game_results['Success']==0],[self.game_results['Predicted']*self.game_results['Over_ML']+(1-self.game_results['Predicted'])*self.game_results['Under_ML'],-1],0)

	def predict_given(self,x_given,y_given,ml_given,info_given):
		x_given = self.trim.transform([x_given])
		self.x_given = self.scaler.transform(x_given)
		self.y_given = self.encoder.transform(y_given.ravel())
		self.ml_given = ml_given
		self.info_given = info_given

		predictions = self.fit.predict(self.x_given)

		if self.ml.shape[1] == 1:
			self.given_results = pd.DataFrame([self.info_given.ravel(),predictions,self.y_given,self.ml_given.ravel(),self.accuracy.ravel()]).T
			self.given_results.columns = ['Matchup','Predicted','Actual','ML','Test_Accuracy']
			self.given_results['Success'] = np.where(self.given_results['Predicted']==self.given_results['Actual'],1,0)
			self.given_results['Payout'] = np.select([self.given_results['Success']==1,self.given_results['Success']==0],[self.given_results['ML'],-1],0)

		else:
			self.given_results = pd.DataFrame([self.info_given.ravel(),predictions,self.y_given,self.ml_given[0].ravel(),self.ml_given[1].ravel(),self.accuracy.ravel()]).T
			self.given_results.columns = ['Matchup','Predicted','Actual','Over_ML','Under_ML','Test_Accuracy']
			self.given_results['Success'] = np.where(self.given_results['Predicted']==self.given_results['Actual'],1,0)
			self.given_results['Payout'] = np.select([self.given_results['Success']==1,self.given_results['Success']==0],[self.given_results['Predicted']*self.given_results['Over_ML']+(1-self.given_results['Predicted'])*self.given_results['Under_ML'],-1],0)

def read_file(team_name,year):
	file_name = team_name.replace(' ','_').replace('.','') + ".csv"

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/' + year + '_Game_Logs')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/' + year + '_Game_Logs')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/' + year + '_Game_Logs')

	data = pd.read_csv(file_name,low_memory=False)
	data.set_index('Date',inplace=True)
	data = data.sort_values('Date')
	data = data.fillna(0)

	return data

def output_vars():
	outs = []

	for stat in ['Win','Hit','Pitch','Over','F5_Over','Cover']:
		outs.append('Overall_30_'+str(stat))
		outs.append('Overall_15_'+str(stat))
		outs.append('Home_Away_'+str(stat))
		outs.append('Right_Left_'+str(stat))
		outs.append('Series_By_Game_'+str(stat))
		outs.append('Betting_Profile_'+str(stat))
		outs.append('Opp_Team_Profile_'+str(stat))
		outs.append('Matchup_Profile_'+str(stat))

		outs.append('Opp_Team_Overall_30_'+str(stat))
		outs.append('Opp_Team_Overall_15_'+str(stat))
		outs.append('Opp_Team_Home_Away_'+str(stat))
		outs.append('Opp_Team_Right_Left_'+str(stat))
		outs.append('Opp_Team_Series_By_Game_'+str(stat))
		outs.append('Opp_Team_Betting_Profile_'+str(stat))
		outs.append('Opp_Team_Opp_Team_Profile_'+str(stat))
		outs.append('Opp_Team_Matchup_Profile_'+str(stat))

		for num in range(1,6):
			outs.append('Overall_'+str(stat)+'_'+str(num))
			outs.append('Home_Away_'+str(stat)+'_'+str(num))
			outs.append('Right_Left_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_Home_Away_'+str(stat)+'_'+str(num))
			outs.append('Series_By_Game_'+str(stat)+'_'+str(num))
			outs.append('Betting_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Profile_'+str(stat)+'_'+str(num))
			outs.append('Matchup_Profile_'+str(stat)+'_'+str(num))
			outs.append('vs_Team_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_Betting_Profile_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_Opp_Team_Profile_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_Matchup_Profile_'+str(stat)+'_'+str(num))
			outs.append('Pitcher_vs_Team_'+str(stat)+'_'+str(num))

			outs.append('Opp_Team_Overall_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Home_Away_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Right_Left_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_Home_Away_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Series_By_Game_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Betting_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Opp_Team_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Team_Matchup_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_Betting_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_Opp_Team_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_Matchup_Profile_'+str(stat)+'_'+str(num))
			outs.append('Opp_Pitcher_vs_Team_'+str(stat)+'_'+str(num))

	return outs

def get_inputs_outputs(df,outcome):
	var_list = output_vars()

	d = {}

	inputs = df[var_list]
	outputs = df[[outcome]]
	info = df[['Matchup']]

	if outcome == 'Win':
		ml = df[['ML']]
	elif outcome == 'Cover':
		ml = df[['RL_ML']]
	else:
		ml = df[['Over_ML','Under_ML']]

	d['Inputs'] = inputs.to_numpy()
	d['Outputs'] = outputs.to_numpy()
	d['Payout'] = ml.to_numpy()
	d['Info'] = info.to_numpy()

	return d

def get_data(file_name,team_name,outcome,date,years,window_size,return_opp_team=None,return_matchups=None):
	d = {}

	for year in years:
		df = read_file(file_name,year)
		df.sort_values('Date',inplace=True)
		dates = list(df.index.unique())
		if return_matchups != None:
			if date in dates:
				matchups = set(df[df.index==date]['Matchup'].to_list())
			else:
				matchups = []
		date_num = dates.index(date)
		start_num = date_num-window_size if date_num>window_size else 0
		if year == '2019':
			end_num = date_num
		else:
			end_num = date_num+window_size if date_num+window_size<len(dates) else -1
		df = df[dates[start_num]:dates[end_num]]
		if team_name == 'All_Teams' and year == '2019':
			df.reset_index(inplace=True)
			df = df.drop(df[df['Date']==date].index)
			df.set_index('Date',inplace=True)
		if return_opp_team != None and year == '2019':
			opp_team_name = df['Opp_Team_Name'][-1]
		var_dict = get_inputs_outputs(df,outcome)
		d.update(var_dict)

	if return_opp_team == None and return_matchups == None:
		return d
	elif return_opp_team != None and return_matchups == None:
		return d,opp_team_name
	elif return_opp_team == None and return_matchups != None:
		return d,matchups
	else:
		return d,opp_team_name,matchups

def model_game(team_name,outcome,date,import_mlb_model=None,import_past_model=None):
	d = {}

	team_data,opp_team_name = get_data(team_name,team_name,outcome,date,['2019'],75,return_opp_team=True)
	model = NeuralNet(team_data['Inputs'],team_data['Outputs'],team_data['Payout'],team_data['Info'])
	model.model()
	model.predict_game()

	opp_team_data = get_data(opp_team_name,opp_team_name,outcome,date,['2019'],75)
	opp_model = NeuralNet(opp_team_data['Inputs'],opp_team_data['Outputs'],opp_team_data['Payout'],opp_team_data['Info'])
	opp_model.model()
	opp_model.predict_game()

	if import_mlb_model == None:
		mlb_data = get_data('All_Teams','All_Teams',outcome,date,['2019'],75)

		mlb_model = NeuralNet(mlb_data['Inputs'],mlb_data['Outputs'],mlb_data['Payout'],mlb_data['Info'])
		mlb_model.model()

		team_mlb_model = deepcopy(mlb_model)
		team_mlb_model.predict_given(team_data['Inputs'][-1],team_data['Outputs'][-1],team_data['Payout'][-1],team_data['Info'][-1])

		opp_team_mlb_model = deepcopy(mlb_model)
		opp_team_mlb_model.predict_given(opp_team_data['Inputs'][-1],opp_team_data['Outputs'][-1],opp_team_data['Payout'][-1],opp_team_data['Info'][-1])

	else:
		team_mlb_model = deepcopy(import_mlb_model)
		team_mlb_model.predict_given(team_data['Inputs'][-1],team_data['Outputs'][-1],team_data['Payout'][-1],team_data['Info'][-1])

		opp_team_mlb_model = deepcopy(import_mlb_model)
		opp_team_mlb_model.predict_given(opp_team_data['Inputs'][-1],opp_team_data['Outputs'][-1],opp_team_data['Payout'][-1],opp_team_data['Info'][-1])

	if import_past_model == None:
		past_data = get_data('All_Teams','All_Teams',outcome,date,['2018'],30)

		past_model = NeuralNet(past_data['Inputs'],past_data['Outputs'],past_data['Payout'],past_data['Info'])
		past_model.model()

		team_past_model = deepcopy(past_model)
		team_past_model.predict_given(team_data['Inputs'][-1],team_data['Outputs'][-1],team_data['Payout'][-1],team_data['Info'][-1])

		opp_team_past_model = deepcopy(past_model)
		opp_team_past_model.predict_given(opp_team_data['Inputs'][-1],opp_team_data['Outputs'][-1],opp_team_data['Payout'][-1],opp_team_data['Info'][-1])

	else:
		team_past_model = deepcopy(import_past_model)
		team_past_model.predict_given(team_data['Inputs'][-1],team_data['Outputs'][-1],team_data['Payout'][-1],team_data['Info'][-1])

		opp_team_past_model = deepcopy(import_past_model)
		opp_team_past_model.predict_given(opp_team_data['Inputs'][-1],opp_team_data['Outputs'][-1],opp_team_data['Payout'][-1],opp_team_data['Info'][-1])


	d['Matchup'] = model.game_results['Matchup'][0]

	if outcome == 'Win' or outcome == 'Cover':
		score = np.sum([2*model.game_results['Predicted'][0],2-2*opp_model.game_results['Predicted'][0],team_mlb_model.given_results['Predicted'][0],1-opp_team_mlb_model.given_results['Predicted'][0],team_past_model.given_results['Predicted'][0],1-opp_team_past_model.given_results['Predicted'][0]])/8

		if score >= 0.8:
			d['Action'] = team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		if score >= 0.7:
			d['Action'] = team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		elif score <= 0.2:
			d['Action'] = opp_team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = opp_model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score <= 0.3:
			d['Action'] = opp_team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = opp_model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		else:
			d['Action'] = 'None'
			d['Bet'] = 0
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.nan
			d['Payout'] = 0

	elif outcome == 'Over' or outcome == 'F5_Over':
		score = np.sum([2*model.game_results['Predicted'][0],2*opp_model.game_results['Predicted'][0],team_mlb_model.given_results['Predicted'][0],opp_team_mlb_model.given_results['Predicted'][0],team_past_model.given_results['Predicted'][0],opp_team_past_model.given_results['Predicted'][0]])/8

		if score >= 0.8:
			d['Action'] = 'Total ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		if score >= 0.7:
			d['Action'] = 'Total ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		elif score <= 0.2:
			d['Action'] = 'Total ' + outcome.replace('_Over',' Under')
			d['Bet'] = 1
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = opp_model.game_results['Under_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score <= 0.3:
			d['Action'] = 'Total ' + outcome.replace('_Over',' Under')
			d['Bet'] = 0.5
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = opp_model.game_results['Under_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		else:
			d['Action'] = 'None'
			d['Bet'] = 0
			d['Confidence'] = np.sum([2*model.game_results['Test_Accuracy'][0],2*opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0],team_past_model.given_results['Test_Accuracy'][0],opp_team_past_model.given_results['Test_Accuracy'][0]])/8
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.nan
			d['Payout'] = 0

	return d

def print_model_game(team_name,outcome,date,silence=None):
	d = model_game(team_name,outcome,date)
	df = pd.DataFrame([d])
	df.set_index('Matchup',inplace=True)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_game_all(team_name,date,silence=None):
	d = []

	for outcome in ['Win','Cover','Over','F5_Over']:
		game = model_game(team_name,outcome,date)
		d.append(game)

	df = pd.DataFrame(d)
	df.set_index('Matchup',inplace=True)

	if silence == None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_date(outcome,date,silence=None):
	d = []

	mlb_data,matchups = get_data('All Teams','All_Teams',outcome,date,['2019'],75,return_matchups=True)

	mlb_model = NeuralNet(mlb_data['Inputs'],mlb_data['Outputs'],mlb_data['Payout'],mlb_data['Info'])
	mlb_model.model()

	past_data = get_data('All Teams','All_Teams',outcome,date,['2018'],30)

	past_model = NeuralNet(past_data['Inputs'],past_data['Outputs'],past_data['Payout'],past_data['Info'])
	past_model.model()

	for matchup in matchups:
		team_name = matchup.split(' @ ')[0]
		game = model_game(team_name,outcome,date,import_mlb_model=mlb_model,import_past_model=past_model)
		d.append(game)
	
	df = pd.DataFrame(d)
	df.set_index('Matchup',inplace=True)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_date_all(date,silence=None):
	d = []

	mlb_df = read_file('All_Teams')
	mlb_df.sort_values('Date',inplace=True)
	mlb_dates = list(mlb_df.index.unique())

	if date in mlb_dates:
		matchups = set(mlb_df[mlb_df.index==date]['Matchup'].to_list())
	else:
		print('No Games Played On ' + date)
		return pd.DataFrame()
	
	for outcome in ['Win','Cover','Over','F5_Over']:
		date_num = mlb_dates.index(date)
		start_num = date_num-75 if date_num>75 else 0
		mlb_df = mlb_df[mlb_dates[start_num]:date]
		mlb_df.reset_index(inplace=True)

		mlb_df = mlb_df.drop(mlb_df[(mlb_df['Date']==date)].index)
		mlb_df.set_index('Date',inplace=True)
		mlb_var_dict = get_inputs_outputs(mlb_df,outcome)

		mlb_model = NeuralNet(mlb_var_dict['Inputs'],mlb_var_dict['Outputs'],mlb_var_dict['Payout'],mlb_var_dict['Info'])
		mlb_model.model()
		
		past_df = read_file('All_Teams','2018')
		past_df.sort_values('Date',inplace=True)

		past_dates = list(past_df.index.unique())
		date_num = past_dates.index(date)
		start_num = date_num-30 if date_num>30 else 0
		end_num = date_num+30 if date_num+30<len(past_dates) else -1
		past_df = past_df[past_dates[start_num]:past_dates[end_num]]

		past_var_dict = get_inputs_outputs(past_df,outcome)

		past_model = NeuralNet(past_var_dict['Inputs'],past_var_dict['Outputs'],past_var_dict['Payout'],past_var_dict['Info'])
		past_model.model()

		for matchup in matchups:
			team_name = matchup.split(' @ ')[0]
			team_df = read_file(team_name)
			game = model_game(team_name,outcome,date,import_mlb_model=mlb_model,import_past_model=past_model)
			d.append(game)
	
	df = pd.DataFrame(d)
	df.set_index('Matchup',inplace=True)
	df.sort_values('Matchup',inplace=True)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_team_season(team_name,outcome,silence=None):
	d = []

	team_df = read_file(team_name)
	dates = team_df.index

	for date in dates[30:]:
		game = model_date(team_name,outcome,date)
		d.append(game)
	
	df = pd.DataFrame(d)
	df.set_index('Matchup',inplace=True)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_team_season_all(team_name,silence=None):
	d = []

	team_df = read_file(team_name)
	dates = team_df.index

	for date in dates[30:]:

		for outcome in ['Win','Cover','Over','F5_Over']:
			game = model_game(team_name,outcome,date)
			d.append(game)
	
	df = pd.DataFrame(d)
	df.set_index('Matchup',inplace=True)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_mlb_season(outcome,silence=None):
	d = []

	mlb_df = read_file('All_Teams','2019')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	for date in dates[30:]:
		try:
			g = model_date(outcome,date,silence=True)
			d.append(g)
			print(date + ' Complete')
		except:
			print(date + ' Not Found')
	
	df = pd.concat(d,axis=0)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_mlb_season_all(silence=None):
	d = []

	mlb_df = read_file('All_Teams','2019')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	for date in dates[30:]:
		for outcome in ['Win','Cover','Over','F5_Over']:
			try:
				g = model_date(outcome,date,silence=True)
				d.append(g)
				print(date + ' ' + outcome + ' Complete')
			except:
				print(date + ' Not Found')
	
	df = pd.concat(d,axis=0)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_date_range(outcome,start_date,end_date,silence=None):
	d = []

	mlb_df = read_file('All_Teams','2019')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	a = dates.index(start_date)
	b = dates.index(end_date)
	dates = dates[a:b]

	for date in dates:
		g = model_date(outcome,date,silence=True)
		d.append(g)
		print(date + ' Complete')
	
	df = pd.concat(d,axis=0)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

def model_date_range_all(start_date,end_date,silence=None):
	d = []

	mlb_df = read_file('All_Teams','2019')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	a = dates.index(start_date)
	b = dates.index(end_date)
	dates = dates[a:b]

	for date in dates:
		for outcome in ['Win','Cover','Over','F5_Over']:
			g = model_date(outcome,date,silence=True)
			d.append(g)
		print(date + ' Complete')
	
	df = pd.concat(d,axis=0)

	if silence==None:
		pd.set_option("display.max_rows", 9999)
		print(df)
		print("")
		print('Accuracy: ' + str(df['Success'].mean()))
		print('Avg. Return: ' + str(df['Payout'].sum()/df['Success'].count()))
		print('Total Profit: ' + str(df['Payout'].sum()))
		print('Num. Bets: ' + str(df['Bet'].sum()))

	return df

##########

