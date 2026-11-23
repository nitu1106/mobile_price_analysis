import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import shap
from lime.lime_tabular import LimeTabularExplainer

def main():
    if not os.path.exists("train.csv"):
        raise FileNotFoundError("Put train.csv beside this Python file.")
    df = pd.read_csv("train.csv")
    if "price_range" not in df: raise ValueError("Target column price_range is missing.")
    df = df.drop(columns=["id"], errors="ignore")
    X = df.drop(columns=["price_range"]).select_dtypes(include="number")
    y = df["price_range"]
    if X.empty: raise ValueError("No numeric feature columns found.")
    feature_names=list(X.columns)
    class_names=[str(c) for c in sorted(y.unique())]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    imputer=SimpleImputer(strategy="median")
    Xtr_arr=imputer.fit_transform(Xtr); Xte_arr=imputer.transform(Xte)
    model=RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1)
    model.fit(Xtr_arr,ytr)
    pred=model.predict(Xte_arr)
    print("Test accuracy:",round(accuracy_score(yte,pred),4))
    print(classification_report(yte,pred))
    out="task66_shap_lime_output"; os.makedirs(out,exist_ok=True)

    # SHAP global feature importance: mean absolute contribution across rows/classes.
    print("Calculating SHAP values...")
    sample=Xte_arr[:min(100,len(Xte_arr))]
    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(sample)
    if isinstance(shap_values,list):
        importance=np.mean([np.abs(v).mean(axis=0) for v in shap_values],axis=0)
    else:
        arr=np.asarray(shap_values)
        if arr.ndim==3:
            if arr.shape[1]==len(feature_names):
                importance=np.abs(arr).mean(axis=(0,2))
            elif arr.shape[2]==len(feature_names):
                importance=np.abs(arr).mean(axis=(0,1))
            else:
                raise ValueError("Unexpected SHAP output shape: "+str(arr.shape))
        elif arr.ndim==2:
            importance=np.abs(arr).mean(axis=0)
        else:
            raise ValueError("Unexpected SHAP output dimensions: "+str(arr.ndim))
    imp_df=pd.DataFrame({"Feature":feature_names,"Mean Absolute SHAP":importance})
    imp_df=imp_df.sort_values("Mean Absolute SHAP",ascending=False)
    imp_df.to_csv(out+"/shap_global_feature_importance.csv",index=False)
    top=imp_df.head(15).sort_values("Mean Absolute SHAP")
    plt.figure(figsize=(9,6)); plt.barh(top["Feature"],top["Mean Absolute SHAP"])
    plt.xlabel("Mean absolute SHAP value"); plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout(); plt.savefig(out+"/shap_global_importance.png",dpi=150); plt.show()

    # SHAP local explanation for first test row.
    one=sample[:1]
    actual=yte.iloc[0]; predicted=model.predict(one)[0]
    print("Local example - actual:",actual,"predicted:",predicted)
    try:
        shap_exp=explainer(one)
        vals=np.asarray(shap_exp.values)
        class_index=list(model.classes_).index(predicted)
        if vals.ndim==3:
            base=np.asarray(shap_exp.base_values)
            base_value=base[0,class_index] if base.ndim>1 else base[class_index]
            local=shap.Explanation(values=vals[0,:,class_index],base_values=base_value,
                                   data=one[0],feature_names=feature_names)
        else:
            base=np.asarray(shap_exp.base_values).reshape(-1)[0]
            local=shap.Explanation(values=vals[0],base_values=base,
                                   data=one[0],feature_names=feature_names)
        shap.plots.waterfall(local,show=False)
        plt.tight_layout(); plt.savefig(out+"/shap_local_waterfall.png",dpi=150,bbox_inches="tight")
        plt.show()
    except Exception as e:
        print("SHAP local waterfall could not be created in this SHAP version:",e)

    # LIME local explanation for the same test row.
    print("Creating LIME explanation...")
    lime_explainer=LimeTabularExplainer(
        training_data=Xtr_arr,feature_names=feature_names,class_names=class_names,
        mode="classification",discretize_continuous=True,random_state=42)
    explanation=lime_explainer.explain_instance(
        data_row=one[0],predict_fn=model.predict_proba,num_features=min(10,len(feature_names)))
    lime_df=pd.DataFrame(explanation.as_list(),columns=["Feature Rule","LIME Contribution"])
    lime_df.to_csv(out+"/lime_local_explanation.csv",index=False)
    fig=explanation.as_pyplot_figure(); plt.tight_layout()
    plt.savefig(out+"/lime_local_explanation.png",dpi=150,bbox_inches="tight"); plt.show()
    print("\\nLIME local explanation:"); print(lime_df.to_string(index=False))
    print("\\nOutputs saved in",out)
    print("SHAP global values summarize feature influence across many rows.")
    print("SHAP and LIME local outputs explain one prediction; they do not prove causation.")

if __name__ == "__main__": main()

# LIME stands for Local Interpretable Model-Agnostic Explanations.

# It is an Explainable AI (XAI) technique used to explain why a machine learning model made a particular prediction.
