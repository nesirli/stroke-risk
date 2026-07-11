import warnings

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from stroke_risk.features.build_features import build_preprocessor
from sklearn.linear_model import LogisticRegression
from stroke_risk.ingest.load_data import load_stroke_data

warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*")

N_TRIALS = 50
optuna.logging.set_verbosity(optuna.logging.WARNING)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_model(X, y):
    def objective_log(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('model', model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study_log = optuna.create_study(direction='maximize', study_name='logreg')
    study_log.optimize(objective_log, n_trials=N_TRIALS)

    print(f'Best log-reg params: {study_log.best_params}')
    print(f'Best log-reg ROC-AUC: {study_log.best_value}')

if __name__ == '__main__':
    X, y = load_stroke_data()
    tune_model(X, y)