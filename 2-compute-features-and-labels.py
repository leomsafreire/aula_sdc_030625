import os
import warnings
import tqdm
import pandas as pd
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab


# Configure file and folder names
datafolder = "data/"
spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")
games = pd.read_hdf(spadl_h5, "games")



xfns = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
    fs.bodypart_onehot,
    fs.result,
    fs.result_onehot,
    fs.goalscore,
    fs.startlocation,
    fs.endlocation,
    fs.movement,
    fs.space_delta,
    fs.startpolar,
    fs.endpolar,
    fs.team,
    fs.time,
    fs.time_delta
]

with pd.HDFStore(spadl_h5) as spadlstore, pd.HDFStore(features_h5) as featurestore:
    for game in tqdm.tqdm(list(games.itertuples()), desc=f"Generating and storing features in {features_h5}"):
        actions = spadlstore[f"actions/game_{game.game_id}"]
        actions = actions.sort_values("action_id").reset_index(drop=True)
        gamestates = fs.gamestates(spadl.add_names(actions), 3)
        
        X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
        featurestore.put(f"game_{game.game_id}", X, format='table')

yfns = [lab.scores, lab.concedes, lab.goal_from_shot]

with pd.HDFStore(spadl_h5) as spadlstore, pd.HDFStore(labels_h5) as labelstore:
    for game in tqdm.tqdm(list(games.itertuples()), desc=f"Computing and storing labels in {labels_h5}"):
        actions = spadlstore[f"actions/game_{game.game_id}"]
        actions = actions.sort_values("action_id").reset_index(drop=True)
        Y = pd.concat([fn(spadl.add_names(actions)) for fn in yfns], axis=1)
        labelstore.put(f"game_{game.game_id}", Y, format='table')
