def get_best_xgb(x_train, y_train):
    """Функция, возвращающая наиболее оптимальную модель XGBClassifier на основе алгоритма Optuna"""
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(param_search_xgb, n_trials=30)
    xgb_best = XGBClassifier(learning_rate=study_xgb.best_params['learning_rate'],
                         max_depth=study_xgb.best_params['max_depth'],
                         n_estimators = study_xgb.best_params['n_estimators'],
                         random_seed=42,
                         subsample=study_xgb.best_params['subsample'],
                         colsample_bylevel= study_xgb.best_params['colsample_bylevel'],
                         gamma=study_xgb.best_params['gamma'])
    return xgb_best
