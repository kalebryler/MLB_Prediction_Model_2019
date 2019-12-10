import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class NeuralNet:
	def __init__(self,x,y,ml,info):
		self.x = x[:-1]
		self.y = y[:-1]
		self.ml = ml[:-1]
		self.info = info[:-1]

		self.x_train,self.x_test,self.y_train,self.y_test,self.ml_train,self.ml_test,self.info_train,self.info_test = train_test_split(self.x,self.y,self.ml,self.info,test_size = 0.25)

		self.x_game = x[-1]
		self.y_game = y[-1]
		self.ml_game = ml[-1]
		self.info_game = info[-1]

		self.scaler = StandardScaler()
		self.scaler.fit(self.x_train)
		self.x_train = self.scaler.transform(self.x_train)
		self.x_test = self.scaler.transform(self.x_test)
		self.x_game = self.scaler.transform([self.x_game])

		self.encoder = preprocessing.LabelEncoder()
		self.encoder.fit(self.y_train.ravel())
		self.y_train = self.encoder.transform(self.y_train.ravel())
		self.y_test = self.encoder.transform(self.y_test.ravel())
		self.y_game = self.encoder.transform(self.y_game.ravel())

		self.input_size = self.x.shape[1]
		self.output_size = len(set(self.y_train))

	def model(self):
		net = MLPClassifier(hidden_layer_sizes=(int(round_up(self.input_size/2,100)),int(round_up(self.input_size/5,100)),int(round_up(self.input_size/10,100))), max_iter=10000)
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
		predictions = self.fit.predict(self.x_game)

		if self.ml.shape[1] == 1:
			self.game_results = pd.DataFrame([self.info_game.ravel(),predictions,self.y_game,self.ml_game.ravel(),self.accuracy.ravel()]).T
			self.game_results.columns = ['Matchup','Predicted','Actual','ML','Test_Accuracy']
			self.game_results['Success'] = np.where(self.game_results['Predicted']==self.game_results['Actual'],1,0)
			self.game_results['Payout'] = np.select([self.game_results['Success']==1,self.game_results['Success']==0],[self.game_results['ML'],-1],0)

		else:
			self.game_results = pd.DataFrame([self.info_game.ravel(),predictions,self.y_game,self.ml_game[0].ravel(),self.ml_game[1].ravel(),self.accuracy.ravel()]).T
			self.game_results.columns = ['Matchup','Predicted','Actual','Over_ML','Under_ML','Test_Accuracy']
			self.game_results['Success'] = np.where(self.game_results['Predicted']==self.game_results['Actual'],1,0)
			self.game_results['Payout'] = np.select([self.game_results['Success']==1,self.game_results['Success']==0],[self.game_results['Predicted']*self.game_results['Over_ML']+(1-self.game_results['Predicted'])*self.game_results['Under_ML'],-1],0)

	def predict_given(self,x_given,y_given,ml_given,info_given):
		self.x_given = self.scaler.transform([x_given])
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

	def predict_given_secondary(self,x_given,y_given,ml_given,info_given):
		self.x_given_secondary = self.scaler.transform(x_given)
		self.y_given_secondary = self.encoder.transform(y_given.ravel())
		self.ml_given_secondary = ml_given[0]
		self.info_given_secondary = info_given[0]

		predictions = self.fit.predict(self.x_given_secondary)

		if self.ml.shape[1] == 1:
			self.given_results_secondary = pd.DataFrame([self.info_given_secondary.ravel(),predictions,self.y_given_secondary,self.ml_given_secondary.ravel(),self.accuracy.ravel()]).T
			self.given_results_secondary.columns = ['Matchup','Predicted','Actual','ML','Test_Accuracy']
			self.given_results_secondary['Success'] = np.where(self.given_results_secondary['Predicted']==self.given_results_secondary['Actual'],1,0)
			self.given_results_secondary['Payout'] = np.select([self.given_results_secondary['Success']==1,self.given_results_secondary['Success']==0],[self.given_results_secondary['ML'],-1],0)

		else:
			self.given_results_secondary = pd.DataFrame([self.info_given_secondary.ravel(),predictions,self.y_given_secondary,self.ml_given_secondary[0].ravel(),self.ml_given_secondary[1].ravel(),self.accuracy.ravel()]).T
			self.given_results_secondary.columns = ['Matchup','Predicted','Actual','Over_ML','Under_ML','Test_Accuracy']
			self.given_results_secondary['Success'] = np.where(self.given_results_secondary['Predicted']==self.given_results_secondary['Actual'],1,0)
			self.given_results_secondary['Payout'] = np.select([self.given_results_secondary['Success']==1,self.given_results_secondary['Success']==0],[self.given_results_secondary['Predicted']*self.given_results_secondary['Over_ML']+(1-self.given_results_secondary['Predicted'])*self.given_results_secondary['Under_ML'],-1],0)


def round_up(x, a):
	return np.ceil(x/a)*a

def round_down(x, a):
	return np.floor(x/a)*a

def read_file(team_name):
	file_name = team_name.replace(' ','_').replace('.','') + ".csv"

	try:
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')
	except:
		os.mkdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')
		os.chdir('/Users/kalebryler/Desktop/MLB_Project/2019_Game_Logs')

	if team_name == 'All_Teams':
		data = pd.read_csv(file_name,low_memory=False)
	else:
		data = pd.read_csv(file_name)
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

def model_game(team_name,outcome,date,import_mlb_model=None,export_mlb_model=False):
	d = {}

	team_df = read_file(team_name)
	team_df = team_df[:date]
	var_dict = get_inputs_outputs(team_df,outcome)
	model = NeuralNet(var_dict['Inputs'],var_dict['Outputs'],var_dict['Payout'],var_dict['Info'])
	model.model()
	model.predict_game()

	opp_team_name = team_df['Opp_Team_Name'][-1]
	opp_team_df = read_file(opp_team_name)
	opp_team_df = opp_team_df[:date]
	opp_var_dict = get_inputs_outputs(opp_team_df,outcome)
	opp_model = NeuralNet(opp_var_dict['Inputs'],opp_var_dict['Outputs'],opp_var_dict['Payout'],opp_var_dict['Info'])
	opp_model.model()
	opp_model.predict_game()

	if import_mlb_model == None:
		mlb_df = read_file('All_Teams')
		mlb_df = mlb_df[:date]
		mlb_df.reset_index(inplace=True)

		team_mlb_df = mlb_df.drop(mlb_df[(mlb_df['Date']==date)&(mlb_df['Team_Name']!=team_name)].index)
		team_mlb_df.set_index('Date',inplace=True)
		team_mlb_var_dict = get_inputs_outputs(team_mlb_df,outcome)

		mlb_model = NeuralNet(team_mlb_var_dict['Inputs'],team_mlb_var_dict['Outputs'],team_mlb_var_dict['Payout'],team_mlb_var_dict['Info'])
		mlb_model.model()

		team_mlb_model = mlb_model
		team_mlb_model.predict_given(var_dict['Inputs'][-1],opp_var_dict['Outputs'][-1],opp_var_dict['Payout'][-1],opp_var_dict['Info'][-1])

		opp_team_mlb_model = mlb_model
		opp_team_mlb_model.predict_given(opp_var_dict['Inputs'][-1],opp_var_dict['Outputs'][-1],opp_var_dict['Payout'][-1],opp_var_dict['Info'][-1])

	else:
		team_mlb_model = import_mlb_model
		team_mlb_model.predict_given(var_dict['Inputs'][-1],opp_var_dict['Outputs'][-1],opp_var_dict['Payout'][-1],opp_var_dict['Info'][-1])

		opp_team_mlb_model = import_mlb_model
		opp_team_mlb_model.predict_given(opp_var_dict['Inputs'][-1],opp_var_dict['Outputs'][-1],opp_var_dict['Payout'][-1],opp_var_dict['Info'][-1])

	d['Matchup'] = model.game_results['Matchup'][0]

	if outcome == 'Win' or outcome == 'Cover':
		score = np.sum([3*model.game_results['Predicted'][0],3-3*opp_model.game_results['Predicted'][0],2*team_mlb_model.given_results['Predicted'][0],2-2*opp_team_mlb_model.given_results['Predicted'][0]])/10

		if score >= 0.8:
			d['Action'] = team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score >= 0.6:
			d['Action'] = team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		elif score <= 0.2:
			d['Action'] = opp_team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = opp_model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score <= 0.4:
			d['Action'] = opp_team_name + ' ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = opp_model.game_results['ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		else:
			d['Action'] = 'None'
			d['Bet'] = 0
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['ML'][0]
			d['Success'] = np.nan
			d['Payout'] = 0

	elif outcome == 'Over' or outcome == 'F5_Over':
		score = np.mean([model.game_results['Predicted'][0],opp_model.game_results['Predicted'][0],team_mlb_model.given_results['Predicted'][0],opp_team_mlb_model.given_results['Predicted'][0]])

		if score >= 0.8:
			d['Action'] = 'Total ' + outcome.replace('_',' ')
			d['Bet'] = 1
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score >= 0.6:
			d['Action'] = 'Total ' + outcome.replace('_',' ')
			d['Bet'] = 0.5
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==1,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		elif score <= 0.2:
			d['Action'] = 'Total ' + outcome.replace('_Over',' Under')
			d['Bet'] = 1
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = opp_model.game_results['Under_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML'],-1).item(0)

		elif score <= 0.4:
			d['Action'] = 'Total ' + outcome.replace('_Over',' Under')
			d['Bet'] = 0.5
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = opp_model.game_results['Under_ML'][0]
			d['Success'] = np.where(model.game_results['Actual'][0]==0,1,0).item(0)
			d['Payout'] = np.where(d['Success']==1,d['ML']/2,-0.5).item(0)

		else:
			d['Action'] = 'None'
			d['Bet'] = 0
			d['Confidence'] = np.mean([model.game_results['Test_Accuracy'][0],opp_model.game_results['Test_Accuracy'][0],team_mlb_model.given_results['Test_Accuracy'][0],opp_team_mlb_model.given_results['Test_Accuracy'][0]])
			d['ML'] = model.game_results['Over_ML'][0]
			d['Success'] = np.nan
			d['Payout'] = 0

	if export_mlb_model == False:
		return d
	else:
		return d, mlb_model

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

	mlb_df = read_file('All_Teams')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	if date in dates:
		matchups = set(mlb_df[mlb_df.index==date]['Matchup'].to_list())
	else:
		print('No Games Played On ' + date)
		return pd.DataFrame()

	mlb_df = mlb_df[:date]
	mlb_df.reset_index(inplace=True)

	mlb_df = mlb_df.drop(mlb_df[mlb_df['Date']==date].index)
	mlb_df.set_index('Date',inplace=True)
	mlb_var_dict = get_inputs_outputs(mlb_df,outcome)

	mlb_model = NeuralNet(mlb_var_dict['Inputs'],mlb_var_dict['Outputs'],mlb_var_dict['Payout'],mlb_var_dict['Info'])
	mlb_model.model()

	for matchup in matchups:
		team_name = matchup.split(' @ ')[0]
		team_df = read_file(team_name)
		game = model_game(team_name,outcome,date,import_mlb_model=mlb_model)
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

def model_date_all(date,silence=None):
	d = []

	mlb_df = read_file('All_Teams')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	if date in dates:
		matchups = set(mlb_df[mlb_df.index==date]['Matchup'].to_list())
	else:
		print('No Games Played On ' + date)
		return pd.DataFrame()
	
	for outcome in ['Win','Cover','Over','F5_Over']:
		mlb_df = mlb_df[:date]
		mlb_df.reset_index(inplace=True)

		mlb_df = mlb_df.drop(mlb_df[mlb_df['Date']==date].index)
		mlb_df.set_index('Date',inplace=True)
		mlb_var_dict = get_inputs_outputs(mlb_df,outcome)

		mlb_model = NeuralNet(mlb_var_dict['Inputs'],mlb_var_dict['Outputs'],mlb_var_dict['Payout'],mlb_var_dict['Info'])
		mlb_model.model()

		for matchup in matchups:
			team_name = matchup.split(' @ ')[0]
			team_df = read_file(team_name)
			game = model_game(team_name,outcome,date,import_mlb_model=mlb_model)
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

	mlb_df = read_file('All_Teams')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	for date in dates[30:]:
		g = model_date(outcome,date,silence=True)
		d.append(g)
		print(date + ' Complete')
	
	df = pd.concat(d,axis=0)
	# df.set_index('Matchup',inplace=True)

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

	mlb_df = read_file('All_Teams')
	mlb_df.sort_values('Date',inplace=True)
	dates = list(mlb_df.index.unique())

	for date in dates[30:]:
		for outcome in ['Win','Cover','Over','F5_Over']:
			g = model_date(outcome,date,silence=True)
			d.append(g)
	
	df = pd.concat(d,axis=0)
	# df.set_index('Matchup',inplace=True)

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
# def print_model(team_name,model_name,outcome):
# 	team_df = read_file(team_name)
# 	var_dict = get_inputs_outputs(team_df)

# 	model = model_name(var_dict[outcome]['Inputs'],var_dict[outcome]['Outputs'],var_dict[outcome]['Payout'])
# 	model.model()
# 	# print(model.metrics)

# 	print(model.results)
# 	print("Error: " + '{:.2%}'.format(model.error))
# 	print("Accuracy: " + '{:.2%}'.format(model.accuracy))
# 	print("Profit: " + '{:.2%}'.format(model.profit))

# def model_team(team_name):
# 	d = {}

# 	team_df = read_file(team_name)
# 	var_dict = get_inputs_outputs(team_df)

# 	for outcome in ['Win','Over','F5_Over','Cover']:
# 		d[outcome] = {}

# 		# d[outcome]['NeuralNet'] = {}
# 		net = NeuralNet(var_dict[outcome]['Inputs'],var_dict[outcome]['Outputs'],var_dict[outcome]['Payout'])
# 		net.model()
# 		# d[outcome]['NeuralNet']['Results'] = net.results
# 		d[outcome]['Error'] = '{:.2%}'.format(net.error)
# 		d[outcome]['Accuracy'] = '{:.2%}'.format(net.accuracy)
# 		d[outcome]['Profit'] = '{:.2%}'.format(net.profit)

# 		# d[outcome]['SVM'] = {}
# 		# svm = SVM(var_dict[outcome]['Inputs'],var_dict[outcome]['Outputs'],var_dict[outcome]['Payout'])
# 		# svm.model()
# 		# # d[outcome]['SVM']['Results'] = svm.results
# 		# d[outcome]['SVM']['Error'] = '{:.2%}'.format(svm.error)
# 		# d[outcome]['SVM']['Accuracy'] = '{:.2%}'.format(svm.accuracy)
# 		# d[outcome]['SVM']['Profit'] = '{:.2%}'.format(svm.profit)

# 	return d

# def model_all():
# 	d = {}

# 	teams = ['All_Teams','St. Louis Cardinals', 'Toronto Blue Jays', 'Los Angeles Angels', 'New York Yankees', 'Arizona Diamondbacks', 'San Diego Padres', 'Atlanta Braves', 'Oakland Athletics', 'Boston Red Sox', 'Cleveland Indians', 'Miami Marlins', 'Colorado Rockies', 'Milwaukee Brewers', 'Houston Astros', 'Minnesota Twins', 'Cincinnati Reds', 'New York Mets', 'Detroit Tigers', 'Philadelphia Phillies', 'Chicago Cubs', 'Seattle Mariners', 'Los Angeles Dodgers', 'San Francisco Giants', 'Pittsburgh Pirates', 'Texas Rangers', 'Chicago White Sox', 'Tampa Bay Rays', 'Kansas City Royals', 'Baltimore Orioles', 'Washington Nationals']

# 	for team in teams:
# 		d[team] = model_team(team)

# 	return d

# def segment(x,y,train_pct):
# 	new_data = np.concatenate((x,y),axis=1)
# 	num = new_data.shape[0]
# 	train_size = int(num*train_pct)
# 	np.random.shuffle(new_data)
# 	train = new_data[:train_size]
# 	test = new_data[train_size:]
# 	x_train = train[:,:x.shape[1]]
# 	x_test = test[:,:x.shape[1]]
# 	y_train = train[:,x.shape[1]:]
# 	y_test = test[:,x.shape[1]:]
# 	return x_train,x_test,y_train,y_test

