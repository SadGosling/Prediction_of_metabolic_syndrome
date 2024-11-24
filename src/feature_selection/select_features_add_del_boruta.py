def select_features_add_del_boruta(x, y):
    """Функция, возвращающая список признаков, отобранных алгоритмами ADD-DEL и Boruta"""
    sfs = SequentialFeatureSelector(
        XGBClassifier(),  # represents the classifier
        k_features=5,
        forward=True,
        floating=True,
        scoring="f1_macro",  # means that the selection will be decided by the accuracy of the classifier.
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
    )
    sfs.fit(x, y)
    df_temp = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    features_add_del = list(df_temp["feature_idx"].values[-1])
    x_add_del = x.iloc[:,features_add_del]
    
    it_imp = IterativeImputer(random_state=42)
    x_imp = it_imp.fit_transform(x)
    model = XGBClassifier()
    feat_selector = BorutaPy(model, n_estimators=100, verbose=1, random_state=42)
    feat_selector.fit(x_imp, y)
    keep = x.columns[feat_selector.support_].to_list()
    keep_ind = pd.DataFrame(x_imp).columns[feat_selector.support_].to_list()
    
    features_selected = list(set([*keep, *x_add_del.columns]))
    return features_selected
