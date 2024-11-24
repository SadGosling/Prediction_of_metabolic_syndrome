def get_best_lgbm(x_train, y_train):
    """Функция, возвращающая наиболее оптимальную модель LGBMClassifier на основе алгоритма Optuna"""
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(param_search_lgbm, n_trials=30)
    lgbm_best = LGBMClassifier(learning_rate=study_lgbm.best_params['learning_rate'],
                             max_depth=study_lgbm.best_params['max_depth'],
                             n_estimators = study_lgbm.best_params['n_estimators'],
                             random_seed=42,
                             subsample=study_lgbm.best_params['subsample'])
    return lgbm_best
