from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
import numpy as np
import optuna
import torch
from optuna import Trial, visualization
from sklearn.metrics import r2_score

def TabNet_fit(X,y,timeout=60*5):

    def Objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 5, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.6, step=0.1)
        n_shared = trial.suggest_int("n_shared", 1, 5)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-8, 1e-3, log=True)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                     lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,seed=999,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=mask_type, n_shared=n_shared,
                     scheduler_params=dict(mode="min",
                                           patience=trial.suggest_int("patienceScheduler",low=3,high=15), # changing sheduler patience to be lower than early stopping patience 
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     ) #early stopping

        kf = KFold(n_splits=5, random_state=42, shuffle=True)
    
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train_1, X_valid = X[train_index], X[test_index]
            y_train_1, y_valid = y[train_index], y[test_index]
            regressor = TabNetRegressor(**tabnet_params)
            regressor.fit(X_train=X_train_1, y_train=y_train_1,
                  eval_set=[(X_valid, y_valid)],
                  patience=trial.suggest_int("patience",low=30,high=100), max_epochs=trial.suggest_int('epochs', 50, 500),
                  eval_metric=['rmse'])
        
            y_pred = regressor.predict(X_valid)
        
            r2 = r2_score(y_valid, y_pred)
        
            CV_score_array.append(-r2)
            avg = np.mean(CV_score_array)

        return avg



    sampler = optuna.samplers.TPESampler(seed=42)  # Using TPE sampler with a fixed seed

    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(Objective, timeout=timeout)

    TabNet_params = study.best_params

    final_params = dict(n_d=TabNet_params['n_da'], n_a=TabNet_params['n_da'], n_steps=TabNet_params['n_steps'], gamma=TabNet_params['gamma'],
                     lambda_sparse=TabNet_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5), seed=999,
                     mask_type=TabNet_params['mask_type'], n_shared=TabNet_params['n_shared'],
                     scheduler_params=dict(mode="max",
                                           patience=TabNet_params['patienceScheduler'],
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     )

    epochs = TabNet_params['epochs']

    regressor = TabNetRegressor(**final_params)
    regressor.fit(X_train=X, y_train=y,
          patience=TabNet_params['patience'], max_epochs=epochs,
          eval_metric=['rmse'])

    return regressor, -study.best_value