from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular
import shap
import os


def get_save_name(estimator_name:str, grid_search:bool|None, outliers_removed:bool):
    # save_name e.g. "random_forest-grid-outlier.sav"
    save_name = estimator_name.replace(" ", "_") 
    if grid_search == None:
        pass
    elif grid_search == True:
        save_name += "-grid"
    else:
        save_name += "-random"

    if outliers_removed:
        save_name += "-outlier"

    save_name += ".sav"
    return save_name

def load_model(estimator_name:str, grid_search:bool, outliers_removed:bool):
    save_name = '../model_tuning/saved_models' + "/" + get_save_name(estimator_name, grid_search, outliers_removed)
    regressor = load(save_name)
    return regressor




def load_scaler():
    return load('fullScaler.sav')


def get_shap_and_lime(model, model_name: str, x, feature_names, sample_indices=[0]):
    save_folder = "explain_saves/"+"_".join(model_name.split(" "))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Explainability analysis for {model_name}")
    max_features = 6 # top 5 + other

    explainer = lime_tabular.LimeTabularExplainer (
        training_data=x,
        feature_names=feature_names,
        class_names=['DITM_IV'],
        mode="regression",
        random_state=69,
        verbose=True
    )   

    print("LIME")
    for i in sample_indices:
        exp = explainer.explain_instance(data_row=x[i], predict_fn=model.predict, num_features=max_features-1)
        exp.save_to_file(f'./{save_folder}/LIME_{model_name}-{i}.html')
        print(exp.as_list())
        fig = exp.as_pyplot_figure()
        fig.savefig(f'./{save_folder}/LIME_{model_name}-{i}.png', bbox_inches='tight')
        plt.clf()

    


    explainer = shap.Explainer(model, x, feature_names=feature_names)
    if model_name in [OLS_NAME, EN_NAME]: # linear explainers
        shap_values = explainer(x)
    else:   
        shap_values = explainer(x, check_additivity=False)
    shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
    plt.savefig(f'./{save_folder}/SHAP_BEE.png', bbox_inches='tight')
    plt.clf()
    shap.plots.bar(shap_values, max_display=6, show=False)
    plt.savefig(f'./{save_folder}/SHAP_BAR.png', bbox_inches='tight')
    plt.clf()
    for i in sample_indices:
        shap.plots.waterfall(shap_values[i], max_display=6, show=False)
        plt.savefig(f'./{save_folder}/SHAP_WF-{i}.png', bbox_inches='tight')
        plt.clf()
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")

RF_NAME = "Random Forest"
HGB_NAME = "Hist Gradient Boosting"
OLS_NAME = "OLS"
EN_NAME = "Elastic Net"

if __name__ == "__main__":
    print("Starting Explainability script")
    df = pd.read_csv("../data/options.csv")
    df = df.sample(frac=0.05, random_state=69)
    feature_names = df.columns.difference(['Unnamed: 0', 'symbol', 'date','DITM_IV'])
    df.drop(columns=['Unnamed: 0', 'symbol', 'date'], inplace=True)
    df.drop(columns=['DITM_IV'], inplace=True)

    scaler = load_scaler() 
    x = scaler.transform(df)

    samples = [0, 2, 20]

    print("Printing samples")
    for i in samples:
        print(i)
        print(x[i])
        print("----------------------------------------------------------")

    # OLS
    model = load_model(OLS_NAME, None, False)
    get_shap_and_lime(model, OLS_NAME, x, feature_names, sample_indices=samples)


    # EN
    model = load_model(EN_NAME, True, False)
    get_shap_and_lime(model, EN_NAME, x, feature_names, sample_indices=samples)



    # HGB
    model = load_model(HGB_NAME, True, False)
    get_shap_and_lime(model, HGB_NAME, x, feature_names, sample_indices=samples)


    # RF
    model = load_model(RF_NAME, True, False)
    print("RF loaded")
    get_shap_and_lime(model, RF_NAME, x, feature_names, sample_indices=samples)






