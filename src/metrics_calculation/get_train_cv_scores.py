def get_train_cv_scores(model, x_train, y_train):
    """Функция, возвращающая значения метрик при кросс-валидации на тренировочной выборке"""
    scoring = ['accuracy','precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    cv_scores = cross_validate(model, x_train, y_train, scoring=scoring)
    metrics = ['test_accuracy', 'test_roc_auc', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    cv_metrics = []
    for metric in metrics: cv_metrics.append(cv_scores[metric])
    cv_metrics.append(f2_score_on_pr_rec(cv_scores['test_precision_macro'], cv_scores['test_recall_macro']))
    cv_metrics = np.array(cv_metrics)
    cv_scores_dict = {'Accuracy':cv_metrics[0],
                'AUC':cv_metrics[1],
                'Precision':cv_metrics[2],
                'Recall':cv_metrics[3],
                'F1-score':cv_metrics[4],
                'F2-score':cv_metrics[5],}
    return cv_scores_dict
