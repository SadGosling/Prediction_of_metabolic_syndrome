def select_features_catboost(x, y,):
    """Функция, возвращающая список признаков, отобранных методом get_feature_importance из CatBoostClassifier"""
    model = CatBoostClassifier(random_seed=42)
    model.fit(
        x,
        y,
        eval_set=(x_test, y_test),
        verbose=200,
        use_best_model=True,
        plot=False,
        early_stopping_rounds=100,
    )
    importances = model.get_feature_importance(catboost.Pool(x_train))
    max_imp = heapq.nlargest(10, importances)
    features_selected_indx = np.where(np.isin(importances, max_imp))[0]
    features_selected = df.columns[features_selected_indx]
    return features_selected
