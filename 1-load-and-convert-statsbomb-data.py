import os
import warnings
import pandas as pd
import tqdm
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")

# Use this if you want to use the free public statsbomb data
# or provide credentials to access the API
SBL = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

# View all available competitions
competitions = SBL.competitions()
set(competitions.competition_name)


selected_competitions = competitions[
    ((competitions.competition_name == "Premier League")
    & (competitions.season_name == "2015/2016"))
    |
    ((competitions.competition_name == "1. Bundesliga")
    & (competitions.season_name == "2015/2016"))
    |
    ((competitions.competition_name == "La Liga")
    & (competitions.season_name == "2015/2016"))
    |
    ((competitions.competition_name == "Serie A")
    & (competitions.season_name == "2015/2016"))
    |
    ((competitions.competition_name == "Ligue 1")
    & (competitions.season_name == "2015/2016"))
]

# Get games from all selected competitions
games = pd.concat([
    SBL.games(row.competition_id, row.season_id)
    for row in selected_competitions.itertuples()
])
games[["home_team_id", "away_team_id", "game_date", "home_score", "away_score"]]

# Load and convert match data
games_verbose = tqdm.tqdm(list(games.itertuples()), desc="Loading game data")
teams, players = [], []
actions = {}
for game in games_verbose:
    # load data
    teams.append(SBL.teams(game.game_id))
    players.append(SBL.players(game.game_id))
    events = SBL.events(game.game_id)
    # convert data
    full_actions = spadl.statsbomb.convert_to_actions(
        events, 
        home_team_id=game.home_team_id,
    )
    
    filtered_actions = full_actions[full_actions['period_id'].isin([1, 2])].reset_index(drop=True)
    filtered_actions = spadl.play_left_to_right(filtered_actions, game.home_team_id)
    actions[game.game_id] = filtered_actions

teams = pd.concat(teams).drop_duplicates(subset="team_id")
players = pd.concat(players)

# Store converted spadl data in a h5-file
datafolder = "data/"

# Create data folder if it doesn't exist
if not os.path.exists(datafolder):
    os.mkdir(datafolder)
    print(f"Directory {datafolder} created.")

spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")

# Store all spadl data in h5-file
with pd.HDFStore(spadl_h5) as spadlstore:
    spadlstore["competitions"] = selected_competitions
    spadlstore["games"] = games
    spadlstore["teams"] = teams
    spadlstore["players"] = players[['player_id', 'player_name', 'nickname']].drop_duplicates(subset='player_id')
    spadlstore["player_games"] = players[['player_id', 'game_id', 'team_id', 'is_starter', 'starting_position_id', 'starting_position_name', 'minutes_played']]
    for game_id in actions.keys():
        spadlstore[f"actions/game_{game_id}"] = actions[game_id]






