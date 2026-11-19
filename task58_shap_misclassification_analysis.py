import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT = 'task58_shap_output'
os.makedirs(OUT, exist_ok=True)
if not os.path.exists('train.csv'):
    raise FileNotFoundError('Put train.csv in the same folder as this script.')
df = pd.read_csv('train.csv')
if 'price_range' not in df.columns:
    raise ValueError("Target column 'price_range' is missing.")
df = df.dropna(subset=['price_range']).copy()
y = df['price_range'].astype(int)
X = df.drop(columns=['price_range'])
if 'id' in X.columns: X = X.drop(columns=['id'])
X = X.select_dtypes(include=np.number)
if X.empty: raise ValueError('No numeric feature columns found.')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))])
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print('Accuracy:', round(accuracy_score(yte, pred), 4))
print('\nClassification report:\n', classification_report(yte, pred))
print('\nConfusion matrix:\n', confusion_matrix(yte, pred))
wrong = pred != yte.to_numpy()
wrong_df = Xte.loc[wrong].copy()
wrong_df.insert(0, 'actual_price_range', yte.loc[wrong].to_numpy())
wrong_df.insert(1, 'predicted_price_range', pred[wrong])
wrong_df.to_csv(os.path.join(OUT, 'misclassified_rows.csv'), index=False)
print(f'\nIncorrect predictions: {wrong.sum()} of {len(yte)}')
if wrong.sum():
    pairs = pd.DataFrame({'actual': yte.to_numpy()[wrong], 'predicted': pred[wrong]})
    pair_counts = pairs.value_counts().reset_index(name='count')
    pair_counts.to_csv(os.path.join(OUT, 'common_error_pairs.csv'), index=False)
    print('\nCommon error pairs:\n', pair_counts.head(10).to_string(index=False))

rf = model.named_steps['rf']
Xready = pd.DataFrame(model.named_steps['imputer'].transform(Xte), columns=Xte.columns, index=Xte.index)
sample = Xready.iloc[:min(100, len(Xready))]
try:
    import shap
    print('\nCalculating SHAP values...')
    vals = shap.TreeExplainer(rf).shap_values(sample)
    if isinstance(vals, list):
        importance = np.mean(np.stack([np.mean(np.abs(v), axis=0) for v in vals]), axis=0)
    else:
        vals = np.asarray(vals)
        if vals.ndim == 3 and vals.shape[1] == sample.shape[1]: importance = np.mean(np.abs(vals), axis=(0, 2))
        elif vals.ndim == 3 and vals.shape[2] == sample.shape[1]: importance = np.mean(np.abs(vals), axis=(0, 1))
        elif vals.ndim == 2: importance = np.mean(np.abs(vals), axis=0)
        else: raise ValueError('Unexpected SHAP output shape.')
    imp = pd.DataFrame({'feature': sample.columns, 'importance': importance}).sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(OUT, 'shap_feature_importance.csv'), index=False)
    title = 'Mean absolute SHAP value'
except Exception as exc:
    print('\nSHAP unavailable; using permutation importance:', exc)
    result = permutation_importance(model, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({'feature': Xte.columns, 'importance': result.importances_mean,
                        'importance_std': result.importances_std}).sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(OUT, 'permutation_importance_fallback.csv'), index=False)
    title = 'Permutation importance (SHAP fallback)'
top = imp.head(15).sort_values('importance')
plt.figure(figsize=(9, 6)); plt.barh(top['feature'], top['importance'])
plt.xlabel('Importance'); plt.title(title); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=150); plt.close()
print('\nFinished. Results saved in:', OUT)
print('Review misclassified_rows.csv and common_error_pairs.csv for recurring mistakes.')


 #SHAP => SHapley Additive exPlanations values explain how much each input feature contributes to a machine learning model's prediction, helping us understand the model's decisions and investigate incorrect predictions.