import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import joblib

Q_PATH = "tables/q_table.pkl"
PLOT_PATH = "./plots/"
PLOT_STATE_PATH = "./plots/single_states/"
REGRESSOR_PATH = "./explainable_shap_models/regressors/"
EXPLAINER_PATH = "./explainable_shap_models/explainers/"
DATASET_PATH = "./explainable_shap_models/dataset/"

N_ESTIMATORS = 300
TRAIN_REGRESSOR = False

BUILD_DATASET = True
BUILD_EXPLAINER = True

SAVE_GLOBAL = True
SAVE_SINGLE = True
IDX = 0  # Index of the state for local explanation


def mkdir(path):
    import os
    if not os.path.exists(path):
        print("Creating directory:", path)
        os.makedirs(path)
        print("Successfully created directory")
    else:
        print("Directory already exists:", path)


def build_dataset(q_dict):
    """
    RockPaperScissors dataset builder for SHAP analysis.
    Input:
        q_dict: dict with keys as history tuples, e.g. (None, None, 0, 1, 1, 2)
                None - round has not been played yet
                0 - Rock, 1 - Paper, 2 - Scissors
                Rightmost values are the most recent rounds.
        qvlas:  values of the Q-table for each action (Rock, Paper, Scissors)
    Output:
        X:  an array that corresponds to the history of states,
            with None replaced by -1
        y0, y1, y2: arrays with Q-values for each action for each state in X.
                    y0 corresponds to Rock
                    y1 to Paper
                    y2 to Scissors
    """
    def sanitize_key(key):
        """
        Converts None values in history to -1.
        """
        return [(-1 if v is None else int(v)) for v in key]

    X, y0, y1, y2 = [], [], [], []
    print(f"Building dataset from Q-table with {len(q_dict)} entries...")
    for key, qvals in q_dict.items():
        x = sanitize_key(key)
        X.append(x)
        y0.append(float(qvals[0]))
        y1.append(float(qvals[1]))
        y2.append(float(qvals[2]))
    X = np.asarray(X)
    print("Dataset built successfully.")
    return X, np.asarray(y0), np.asarray(y1), np.asarray(y2)


def save_dataset(X, y0, y1, y2, feature_names, folder, filename):
    import os
    import numpy as np

    np.savez(os.path.join(folder, filename), X=X, y0=y0, y1=y1, y2=y2,
             feature_names=np.array(feature_names, dtype=object),
             allow_pickle=True)


def expand_history_onehot(X):
    # X: shape (n_samples, 2*slots), values in {-1,0,1,2} where -1 means empty
    n, cols = X.shape
    slots = cols // 2
    feats = []
    feature_names = []
    for i in range(slots):
        for p in (1, 2):
            col = X[:, i*2 + (p-1)]
            is_rock = (col == 0).astype(int)[:, None]
            is_paper = (col == 1).astype(int)[:, None]
            is_scissors = (col == 2).astype(int)[:, None]
            is_empty = (col == -1).astype(int)[:, None]
            feats.append(is_rock)
            feats.append(is_paper)
            feats.append(is_scissors)
            feats.append(is_empty)
            feature_names += [
                f"h{i}_p{p}_rock",
                f"h{i}_p{p}_paper",
                f"h{i}_p{p}_scissors",
                f"h{i}_p{p}_empty",
            ]
    X_expanded = np.hstack(feats)
    return X_expanded, feature_names


def load_dataset(folder, filename):
    import os
    import numpy as np

    data = np.load(os.path.join(folder, filename), allow_pickle=True)
    return data["X"], data["y0"], data["y1"], data["y2"], \
        data["feature_names"].tolist()


def load_q_table(q_path):
    import pickle
    print(f"Loading Q-table from '{q_path}'...")
    with open(q_path, "rb") as f:
        q_dict = pickle.load(f)
    print("Q-table loaded successfully.")
    return q_dict


def save_global_shap(shap_values, features, feature_names, folder, filename):
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=features,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{folder}{filename}")
    plt.close()


def save_single_shap(explainer, idx, shap_values, features,
                     feature_names, folder, filename):
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(10, 6))
    shap.force_plot(explainer.expected_value, shap_values[idx], features[idx],
                    feature_names=feature_names, show=False, matplotlib=True)
    plt.tight_layout()
    plt.savefig(f"{folder}{filename}")
    plt.close()


def explain_shap(PLOT_PATH, REGRESSOR_PATH, EXPLAINER_PATH, DATASET_PATH,
                 Q_PATH, PLOT_STATE_PATH, BUILD_DATASET, BUILD_EXPLAINER,
                 N_ESTIMATORS, TRAIN_REGRESSOR, IDX):

    mkdir(PLOT_PATH)
    mkdir(PLOT_STATE_PATH)
    mkdir(REGRESSOR_PATH)
    mkdir(EXPLAINER_PATH)
    mkdir(DATASET_PATH)

    if BUILD_DATASET:
        print("Building dataset...")
        q_dict = load_q_table(Q_PATH)
        X, y0, y1, y2 = build_dataset(q_dict)
        X, feature_names = expand_history_onehot(X)
        save_dataset(X, y0, y1, y2, feature_names,
                     DATASET_PATH, "rps_dataset.npz")
        print("Dataset built successfully.")
    else:
        print("Loading dataset...")
        X, y0, y1, y2, feature_names = load_dataset(DATASET_PATH,
                                                    "rps_dataset.npz")
        print("Dataset loaded successfully.")

    # Regressors for each action
    if TRAIN_REGRESSOR:
        print("Training regressors...")
        rock_regressor = RandomForestRegressor(N_ESTIMATORS,
                                               random_state=42).fit(X, y0)
        paper_regressor = RandomForestRegressor(N_ESTIMATORS,
                                                random_state=42).fit(X, y1)
        scissors_regressor = RandomForestRegressor(N_ESTIMATORS,
                                                   random_state=42).fit(X, y2)
        joblib.dump(rock_regressor,
                    f"{REGRESSOR_PATH}rock_regressor.pkl")
        joblib.dump(paper_regressor,
                    f"{REGRESSOR_PATH}paper_regressor.pkl")
        joblib.dump(scissors_regressor,
                    f"{REGRESSOR_PATH}scissors_regressor.pkl")
        print("Regressors trained and saved successfully.")

    else:
        print("Loading trained regressors...")
        rock_regressor = joblib.load(
            f"{REGRESSOR_PATH}rock_regressor.pkl")
        paper_regressor = joblib.load(
            f"{REGRESSOR_PATH}paper_regressor.pkl")
        scissors_regressor = joblib.load(
            f"{REGRESSOR_PATH}scissors_regressor.pkl")
        print("Loaded regressors successfully.")

    # Explainers for each action
    if BUILD_EXPLAINER:
        print("Building explainers and calculating SHAP values...")
        rock_explainer = shap.TreeExplainer(rock_regressor)
        rock_shap_values = rock_explainer.shap_values(X)
        joblib.dump(rock_explainer,
                    f"{EXPLAINER_PATH}rock_explainer.pkl")
        joblib.dump(rock_shap_values,
                    f"{EXPLAINER_PATH}rock_shap_values.pkl")
        print("Rock explainer and values saved successfully.")

        paper_explainer = shap.TreeExplainer(paper_regressor)
        paper_shap_values = paper_explainer.shap_values(X)
        joblib.dump(paper_explainer,
                    f"{EXPLAINER_PATH}paper_explainer.pkl")
        joblib.dump(paper_shap_values,
                    f"{EXPLAINER_PATH}paper_shap_values.pkl")
        print("Paper explainer and values saved successfully.")

        scissors_explainer = shap.TreeExplainer(scissors_regressor)
        scissors_shap_values = scissors_explainer.shap_values(X)
        joblib.dump(scissors_explainer,
                    f"{EXPLAINER_PATH}scissors_explainer.pkl")
        joblib.dump(scissors_shap_values,
                    f"{EXPLAINER_PATH}scissors_shap_values.pkl")
        print("Scissors explainer and values saved successfully.")
    else:
        print("Loading trained explainers...")
        rock_explainer = joblib.load(
            f"{EXPLAINER_PATH}rock_explainer.pkl")
        paper_explainer = joblib.load(
            f"{EXPLAINER_PATH}paper_explainer.pkl")
        scissors_explainer = joblib.load(
            f"{EXPLAINER_PATH}scissors_explainer.pkl")
        print("Loaded explainers successfully.")

        print("Loading SHAP values...")
        obj_rock = joblib.load(
            f"{EXPLAINER_PATH}rock_shap_values.pkl")
        obj_paper = joblib.load(
            f"{EXPLAINER_PATH}paper_shap_values.pkl")
        obj_scissors = joblib.load(
            f"{EXPLAINER_PATH}scissors_shap_values.pkl")
        rock_shap_values = obj_rock
        paper_shap_values = obj_paper
        scissors_shap_values = obj_scissors
        print("SHAP values loaded successfully.")

    # Global plots
    if SAVE_GLOBAL:
        print("Saving global SHAP summary plots...")
        save_global_shap(rock_shap_values, X, feature_names,
                         PLOT_PATH, filename="rock_global.png")
        save_global_shap(paper_shap_values, X, feature_names,
                         PLOT_PATH, filename="paper_global.png")
        save_global_shap(scissors_shap_values, X, feature_names,
                         PLOT_PATH, filename="scissors_global.png")
        print("Global SHAP summary plots saved successfully.")

    # Local explanation for a single state
    if SAVE_SINGLE:
        print("Saving local SHAP explanation plots for a single state...")
        save_single_shap(rock_explainer, IDX, rock_shap_values, X,
                         feature_names, PLOT_STATE_PATH,
                         f"rock_single_state_{IDX}.png")
        save_single_shap(paper_explainer, IDX, paper_shap_values, X,
                         feature_names, PLOT_STATE_PATH,
                         f"paper_single_state_{IDX}.png")
        save_single_shap(scissors_explainer, IDX, scissors_shap_values, X,
                         feature_names, PLOT_STATE_PATH,
                         f"scissors_single_state_{IDX}.png")
        print("Single state SHAP explanations saved successfully.")


if __name__ == "__main__":
    BUILD_DATASET = False
    BUILD_EXPLAINER = False

    N_ESTIMATORS = 10
    TRAIN_REGRESSOR = False

    SAVE_GLOBAL = False
    SAVE_SINGLE = True
    IDX = 35

    explain_shap(PLOT_PATH, REGRESSOR_PATH, EXPLAINER_PATH, DATASET_PATH,
                 Q_PATH, PLOT_STATE_PATH, BUILD_DATASET, BUILD_EXPLAINER,
                 N_ESTIMATORS, TRAIN_REGRESSOR, IDX)
