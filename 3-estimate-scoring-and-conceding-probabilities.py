import os
import warnings
import tqdm
import pandas as pd
import xgboost
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Configure file and folder names
datafolder = "data"
spadl_h5 = os.path.join(datafolder, "spadl-statsbomb.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")
predictions_h5 = os.path.join(datafolder, "predictions.h5")

games = pd.read_hdf(spadl_h5, "games")
print("nb of games:", len(games))

traingames = games[games['competition_id'] != 2]
testgames = games[games['competition_id']==2]

# 1. Select feature set X
xfns = [
    fs.actiontype,
    fs.actiontype_onehot,
    #fs.bodypart,
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
    #fs.time,
    fs.time_delta,
    #fs.actiontype_result_onehot
]
nb_prev_actions = 3

Xcols = fs.feature_column_names(xfns, nb_prev_actions)

def getXY(games,Xcols):
    # generate the columns of the selected feature
    X = []
    for game_id in tqdm.tqdm(games.game_id, desc="Selecting features"):
        Xi = pd.read_hdf(features_h5, f"game_{game_id}")
        X.append(Xi[Xcols])
    X = pd.concat(X).reset_index(drop=True)

    # 2. Select label Y
    Ycols = ["scores","concedes"]
    Y = []
    for game_id in tqdm.tqdm(games.game_id, desc="Selecting label"):
        Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
        Y.append(Yi[Ycols])
    Y = pd.concat(Y).reset_index(drop=True)
    return X, Y

X, Y = getXY(traingames,Xcols)
print("X:", list(X.columns))
print("Y:", list(Y.columns))

# Train classifiers F(X) = Y
Y_hat = pd.DataFrame()
models = {}

for col in list(Y.columns):
    model = xgboost.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        n_jobs=-3,
        verbosity=1,
        enable_categorical=True
    )
    model.fit(X, Y[col])
    models[col] = model

# Evaluate the model
testX, testY = getXY(testgames,Xcols)

def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

for col in testY.columns:
    Y_hat[col] = [p[1] for p in models[col].predict_proba(testX)]
    print(f"### Y: {col} ###")
    evaluate(testY[col], Y_hat[col])


A = []
for game_id in tqdm.tqdm(testgames.game_id, "Loading game ids"):
    Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
    A.append(Ai[["game_id"]])
A = pd.concat(A)
A = A.reset_index(drop=True)

grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
with pd.HDFStore(predictions_h5) as predictionstore:
    for k, df in tqdm.tqdm(grouped_predictions, desc="Saving predictions per game"):
        df = df.reset_index(drop=True)
        predictionstore.put(f"game_{int(k)}", df[Y_hat.columns])
