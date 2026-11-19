import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

OUT = 'task60_evaluation_output'
os.makedirs(OUT, exist_ok=True)
if not os.path.exists('train.csv'): raise FileNotFoundError('Put train.csv in the same folder as this script.')
df = pd.read_csv('train.csv')
if 'price_range' not in df.columns: raise ValueError("Target column 'price_range' is missing.")
df = df.dropna(subset=['price_range']).copy()
y = df['price_range'].astype(int)
X = df.drop(columns=['price_range'])
if 'id' in X.columns: X = X.drop(columns=['id'])
X = X.select_dtypes(include=np.number)
if X.empty: raise ValueError('No numeric feature columns found.')
models = {
    'Random Forest': Pipeline([('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))]),
    'Extra Trees': Pipeline([('imputer', SimpleImputer(strategy='median')),
        ('model', ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1))]),
    'SVM RBF': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))])
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 'precision_weighted': 'precision_weighted',
           'recall_weighted': 'recall_weighted', 'f1_weighted': 'f1_weighted'}
rows = []
for name, model in models.items():
    print('Evaluating:', name)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    rows.append({'model': name, 'accuracy_mean': scores['test_accuracy'].mean(),
        'accuracy_std': scores['test_accuracy'].std(),
        'precision_weighted_mean': scores['test_precision_weighted'].mean(),
        'recall_weighted_mean': scores['test_recall_weighted'].mean(),
        'f1_weighted_mean': scores['test_f1_weighted'].mean(),
        'f1_weighted_std': scores['test_f1_weighted'].std(),
        'average_fit_time_seconds': scores['fit_time'].mean()})
summary = pd.DataFrame(rows).sort_values('f1_weighted_mean', ascending=False)
summary.to_csv(os.path.join(OUT, 'model_evaluation_summary.csv'), index=False)
best = summary.iloc[0]
class_counts = y.value_counts().sort_index()
lines = ['MOBILE PRICE ANALYSIS - MODEL EVALUATION REPORT', '=' * 55, '', 'DATASET OVERVIEW',
    f'Rows used: {len(df)}', f'Numeric features: {X.shape[1]}',
    f'Feature names: {", ".join(map(str, X.columns))}',
    f'Missing feature values: {int(X.isna().sum().sum())}', f'Duplicate rows: {int(df.duplicated().sum())}', '',
    'TARGET CLASS DISTRIBUTION']
for label, count in class_counts.items(): lines.append(f'Price range {label}: {count} rows')
lines += ['', 'MODEL COMPARISON - 5-FOLD STRATIFIED CROSS-VALIDATION', summary.to_string(index=False), '',
    f"Best model by mean weighted F1: {best['model']}",
    f"Mean weighted F1: {best['f1_weighted_mean']:.4f}",
    f"Weighted F1 standard deviation: {best['f1_weighted_std']:.4f}", '', 'IMPROVEMENT PLAN',
    '1. Data quality: inspect missing values, duplicate rows, invalid values and class balance.',
    '2. Feature engineering: test useful combinations of existing mobile features.',
    '   Examples if columns exist: pixel count = px_height x px_width; screen area = sc_h x sc_w;',
    '   battery capacity per unit weight = battery_power / weight.',
    '3. Feature importance: use SHAP or permutation importance to identify influential features.',
    '4. Hyperparameter tuning: use GridSearchCV or RandomizedSearchCV on the best candidates.',
    '   For trees tune n_estimators, max_depth, min_samples_split and max_features.',
    '   For SVM tune C and gamma; keep scaling inside the pipeline.',
    '5. Algorithm selection: compare tree models, SVM, Gradient Boosting and XGBoost fairly.',
    '6. Generalization: reserve a final untouched test set and evaluate after model selection.',
    '7. Probability calibration: compare sigmoid and isotonic if confidence scores matter.', '',
    'INTERPRETATION', '- Higher accuracy, precision, recall and F1 are generally better.',
    '- Lower cross-validation F1 standard deviation suggests more stable performance.',
    '- Select metrics based on the task goal; do not rely on accuracy alone.',
    '- Cross-validation is an estimate; confirm the final choice on an untouched test set.', '',
    'See model_evaluation_summary.csv for the numeric comparison.']
report = os.path.join(OUT, 'model_evaluation_report.txt')
with open(report, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
print('\nModel comparison:\n', summary.to_string(index=False))
print('\nBest model:', best['model']); print('Report saved to:', report)
