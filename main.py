import scrape_data
import learning_model

def read_team_name(team_name=None):
	team_map_1 = {'STL':'St. Louis Cardinals','TOR':'Toronto Blue Jays','LAA':'Los Angeles Angels','NYY':'New York Yankees','ARI':'Arizona Diamondbacks','SD':'San Diego Padres','ATL':'Atlanta Braves','OAK':'Oakland Athletics','BOS':'Boston Red Sox','CLE':'Cleveland Indians','MIA':'Miami Marlins','COL':'Colorado Rockies','MIL':'Milwaukee Brewers','HOU':'Houston Astros','MIN':'Minnesota Twins','CIN':'Cincinnati Reds','NYM':'New York Mets','DET':'Detroit Tigers','PHI':'Philadelphia Phillies','CHC':'Chicago Cubs','SEA':'Seattle Mariners','LAD':'Los Angeles Dodgers','SF':'San Francisco Giants','PIT':'Pittsburgh Pirates','TEX':'Texas Rangers','CWS':'Chicago White Sox','TB':'Tampa Bay Rays','KC':'Kansas City Royals','BAL':'Baltimore Orioles','WSH':'Washington Nationals'}
	
	if team_name is not None:
		if team_name in team_map_1:
			return team_map_1[team_name]
		else:
			team_name = team_name.title()
			return team_name
	else:
		return 0

def read_bet_type(bet_type=None):
	if bet_type is not None:
		bet_type = bet_type.title()
		bet_type = bet_type.replace(' ','_')
		return bet_type
	else:
		return 0

def read_date(date=None):
	if date is not None:
		if date[0] == '0' or date[0] == '1':
			return date
		else:
			date = '0' + date
			return date
	else:
		return 0

def main():

	while True:
		print("")
		print("Menu Options")
		print("")
		print("1. Update Database")
		print("2. Model Game")
		print("3. Model Date")
		print("4. Model Team's Season")
		print("5. Model MLB Season")
		print("6. Exit")
		print("")

		user_choice = input("Select Option Number: ")
		print("")

		if user_choice == '1':
			scrape_data.write_game_logs()
			print("")
			continue

		elif user_choice == '2':
			team_name = read_team_name(input("Team Name: "))
			outcome = read_bet_type(input("Bet Type (Win / Cover / Over / F5 Over / All): "))
			date = read_date(input("Date (Month / Day): "))
			print("")

			if outcome == 'All':
				try:
					output = learning_model.model_game_all(team_name,date)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue
			else:
				try:
					output = learning_model.print_model_game(team_name,outcome,date)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue

		elif user_choice == '3':
			outcome = read_bet_type(input("Bet Type (Win/Cover/Over/F5 Over/All): "))
			date = read_date(input("Date (Month/Day): "))
			print("")

			if outcome == 'All':
				try:
					output = learning_model.model_date_all(date)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue
			else:
				try:
					output = learning_model.model_date(outcome,date)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue


		elif user_choice == '4':
			team_name = read_team_name(input("Team Name: "))
			outcome = read_bet_type(input("Bet Type (Win/Cover/Over/F5 Over/All): "))
			print("")

			if outcome == 'All':
				try:
					output = learning_model.model_team_season_all(team_name)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue
			else:
				try:
					output = learning_model.model_team_season(team_name,outcome)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue

		elif user_choice == '5':
			outcome = read_bet_type(input("Bet Type: "))
			print("")

			if outcome == 'All':
				try:
					output = learning_model.model_mlb_season_all()
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue
			else:
				try:
					output = learning_model.model_mlb_season(outcome)
					print("")
					continue
				except:
					print("Invalid Input.")
					print("")
					continue

		elif user_choice == '6':
			break

		# elif user_choice == '7':
		# 	team_name = 'Texas Rangers'
		# 	outcome = 'F5_Over'
		# 	date = '07/23'

		# 	output = learning_model.model_date_all(date)
		# 	print("")
		# 	continue

		else:
			print("Invalid Input")
			print("")
			continue

	return

main()