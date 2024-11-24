def get_best_rfc(x_train_imp, y_train):
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(param_search_rfc, n_trials=30)
    rfc_best = RandomForestClassifier(n_estimators = study_rf.best_params['n_estimators'],
                                      max_depth = study_rf.best_params['max_depth'],
                                      min_samples_split = study_rf.best_params['min_samples_split'],
                                      min_samples_leaf = study_rf.best_params['min_samples_leaf'],
                                      criterion='log_loss')
    return rfc_best
