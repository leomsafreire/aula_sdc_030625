import os
import warnings
import tqdm
import pandas as pd
import socceraction.spadl as spadl
import socceraction.vaep.formula as vaepformula


datafolder = "data"
spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
predictions_h5 = os.path.join(datafolder, "predictions.h5")

with pd.HDFStore(spadl_h5) as spadlstore:
    games = (
        spadlstore["games"]
        .merge(spadlstore["competitions"], how='left')
        .merge(spadlstore["teams"].add_prefix('home_'), how='left')
        .merge(spadlstore["teams"].add_prefix('away_'), how='left')
    )
    players = spadlstore["players"]
    teams = spadlstore["teams"]


games = games[games["competition_id"] == 2].reset_index(drop=True)

A = []
for game in tqdm.tqdm(list(games.itertuples()), desc="Rating actions"):
    actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
    actions = (
        spadl.add_names(actions)
        .merge(players, how="left")
        .merge(teams, how="left")
        .sort_values(["game_id", "period_id", "action_id"])
        .reset_index(drop=True)
    )
    preds = pd.read_hdf(predictions_h5, f"game_{game.game_id}")
    values = vaepformula.value(actions, preds.scores, preds.concedes)
    A.append(pd.concat([actions, preds, values], axis=1))

A = pd.concat(A).sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

# Save valued actions to H5 file
valued_actions_h5 = os.path.join(datafolder, "valued_actions.h5")
A.to_hdf(valued_actions_h5, key="valued_actions", mode="w")
print(f"Saved valued actions to {valued_actions_h5}")



