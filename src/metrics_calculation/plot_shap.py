def plot_shap(model, x_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_train)
    shap.plots.bar(shap_values)
