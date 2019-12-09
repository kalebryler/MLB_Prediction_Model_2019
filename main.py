import scrape_data
import learning_model

def read_team_name(team_name):
	team_name = team_name.title()
	return team_name

def read_bet_type(bet_type):
	bet_type = bet_type.title()
	bet_type = bet_type.replace(' ','_')
	return bet_type

def read_date(date):
	if date[0] == '0' or date[0] == '1':
		return date
	else:
		date = '0' + date
		return date

def main():

	while True:
		print("Menu Options")
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
			continue

		elif user_choice == '2':
			team_name = read_team_name(input("Team Name: "))
			outcome = read_bet_type(input("Bet Type: "))
			date = read_date(input("Date: "))
			print("")

			try:
				output = learning_model.print_model_game(team_name,outcome,date)
				print("")
				continue
			except:
				print("Invalid Input.")
				print("")
				continue

		elif user_choice == '3':
			outcome = read_bet_type(input("Bet Type: "))
			date = read_date(input("Date: "))
			print("")

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
			outcome = read_bet_type(input("Bet Type: "))
			print("")

			try:
				output = learning_model.model_team_season(team_name,outcome)
				print("")
				continue
			except:
				print("Invalid Input")
				print("")
				continue

		elif user_choice == '5':
			outcome = read_bet_type(input("Bet Type: "))
			print("")

			try:
				output = learning_model.model_mlb_season(outcome)
				print("")
				continue
			except:
				print("Invalid Input")
				print("")
				continue

		elif user_choice == '6':
			break

		elif user_choice == '7':
			team_name = 'Texas Rangers'
			outcome = 'Over'
			date = '07/32'

			output = learning_model.print_model_game(team_name,outcome,date)
			print("")
			continue

		else:
			print("Invalid Input")
			print("")
			continue


	return

main()