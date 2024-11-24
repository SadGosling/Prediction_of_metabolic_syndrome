def get_test_scores(model, x_test, y_test):
  """Функция, возвращающая значения метрик на переданной выборке"""
  y_pred = model.predict(x_test)
  round_num = 2
  result_dict = {
      "Accuracy": round(accuracy_score(y_test, y_pred), round_num),
      "AUC": round(roc_auc_score(y_test, y_pred), round_num),
      "Precision": round(precision_score(y_test, y_pred), round_num),
      "Recall": round(recall_score(y_test, y_pred), round_num),
      "F1-score": round(f1_score(y_test, y_pred), round_num),
      "F2-score": round(f2_score(y_test, y_pred), round_num)
  }
  return result_dict
