def param_search_xgb_rfc(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
    }

    model = XGBRFClassifier(**params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return f1_score(y_test, predictions)
