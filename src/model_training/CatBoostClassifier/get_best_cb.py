def get_best_cb(x_train, y_train):
    """Функция, возвращающая наиболее оптимальную модель CatBoostClassifier на основе алгоритма Optuna"""
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(param_search_cb, n_trials=30)
    cb_best = CatBoostClassifier(learning_rate=study_cb.best_params['learning_rate'],
                             depth=study_cb.best_params['depth'],
                             random_seed=42,
                             subsample=study_cb.best_params['subsample'],
                             colsample_bylevel=study_cb.best_params['colsample_bylevel'],
                             min_data_in_leaf =study_cb.best_params['min_data_in_leaf'],
                             random_strength=study_cb.best_params['random_strength'])
    return cb_best
