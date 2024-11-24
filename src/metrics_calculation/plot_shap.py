def plot_shap(model, x_train):
    """Функция, рисующая средние значения SHAP для переданной модели на датасете"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_train)
    shap.plots.bar(shap_values)
