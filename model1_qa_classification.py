"""
=============================================================
Model 1 — QA Pass/Fail Prediction
Dataset: Per_protocol.xlsx
  PASS (1): ' GPR' sheet, Status=ΕΠΙΤΥΧΙΑ
  FAIL (0): ' GPR' sheet, Status=ΑΠΟΤΥΧΙΑ + ' Fail QA' sheet

NOTE: N per group is printed at runtime, not hardcoded.

Features: 12 complexity metrics + protocol + anatomical site
Models: Random Forest + XGBoost

BUG FIX: the Fail QA rows' protocol used to be labeled 'Unknown'
(a sentinel value that never appears among PASS rows), which let
the model shortcut to "Protocol == Unknown -> FAIL". Since dosisoft
(2%/2mm, global) is always attempted first at this center, Fail QA
rows are now labeled 'dosisoft' — the real protocol that was tried.
=============================================================
"""

# ================================================================
#  IMPORTANT — NOT FOR GENERAL USE
# ================================================================
# This script is provided for REFERENCE and reproducibility of the
# results presented in this thesis. It requires the labelled dataset
# 'Per_protocol.xlsx' which contains patient-related clinical data
# and CANNOT be shared due to confidentiality.
#
# This script CANNOT be executed without the corresponding dataset.
# It is NOT intended as a general-purpose tool for other datasets.
#
# For reusable code, see:
#   - all_metrics.py   (VMAT complexity extraction from DICOM-RT)
#   - csv_to_excel.py  (CSV to Excel converter)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import openpyxl
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("️  XGBoost not found. Install with: pip install xgboost")
    HAS_XGB = False


# ============================================================
#  USER SETTINGS — CHANGE THIS PATH FOR YOUR SYSTEM
# ============================================================

FILE = "Per_protocol.xlsx"  # Change to full path if needed

# ============================================================


# =============================================================
# 1. DATA LOADING
# =============================================================
print("=" * 60)
print(" MODEL 1 — QA PASS/FAIL PREDICTION")
print("=" * 60)

METRIC_COLS = [
    'MCSv', 'Total_MU', 'LSV_Mean', 'AAV_Mean', 'MI_Mean',
    'Leaf_Travel_mm', 'LTMCS', 'MU_per_CP_Mean', 'MU_lt_3_Percent',
    'SAS5_Mean', 'SAS10_Mean', 'AAJA_Mean'
]

def norm_proto(p):
    """Normalize protocol name to 4 categories."""
    if not p or str(p).strip() in ('', 'nan', 'Unknown'): return 'Unknown'
    p = str(p).replace('\xa0', ' ').lower().strip()
    if 'dosisoft' in p: return 'dosisoft'
    if '90%' in p:      return '3mm90'
    if '3% 2mm' in p or '3%2mm' in p: return '3mm2'
    if '3% 3mm' in p or '3%3mm' in p: return '3mm3'
    return 'Unknown'

def norm_site(d):
    """Map diagnosis to anatomical site group."""
    if not d: return 'Unknown'
    d = str(d).upper()
    if 'PROSTATE' in d:                                                return 'Prostate'
    if any(x in d for x in ['ENDOMETRIUM','CERVIX','VULVA','OVARY']):  return 'Gynecological'
    if any(x in d for x in ['BLADDER','RECTUM','ANUS']):               return 'Pelvic'
    return 'Other'

# Robust sheet lookup (handles inconsistent leading/trailing spaces)
wb = openpyxl.load_workbook(FILE, read_only=True)
sheet_map = {s.strip().lower(): s for s in wb.sheetnames}

def find_sheet(target):
    key = target.strip().lower()
    if key not in sheet_map:
        raise ValueError(f"Could not find sheet matching '{target}'. Sheets: {wb.sheetnames}")
    return sheet_map[key]

GPR_SHEET  = find_sheet('GPR')
FAIL_SHEET = find_sheet('Fail QA')

# ── Source 1: GPR sheet (PASS and FAIL per protocol) ────────
df_gpr = pd.read_excel(FILE, sheet_name=GPR_SHEET)
df_gpr['QA_Label'] = df_gpr['Status'].map({'ΕΠΙΤΥΧΙΑ': 1, 'ΑΠΟΤΥΧΙΑ': 0})
df_gpr['Protocol'] = df_gpr['Πρωτόκολλο'].apply(norm_proto)
df_gpr['Site']     = df_gpr['Διάγνωση'].apply(norm_site)
df_gpr_clean = df_gpr[METRIC_COLS + ['Protocol','Site','QA_Label']].dropna(subset=['QA_Label'])

# ── Source 2: Fail QA sheet (failed ALL protocols) ──────────
df_fail = pd.read_excel(FILE, sheet_name=FAIL_SHEET)
df_fail['Site']     = df_fail['Διάγνωση (filename)'].apply(norm_site)
df_fail['Protocol'] = 'dosisoft'  # dosisoft (2%/2mm) is always tried first
df_fail['QA_Label'] = 0
df_fail_clean = df_fail[METRIC_COLS + ['Protocol','Site','QA_Label']]

# ── Combine ──────────────────────────────────────────────────
df = pd.concat([df_gpr_clean, df_fail_clean], ignore_index=True)
df = df.dropna(subset=METRIC_COLS)

n_pass = (df['QA_Label'] == 1).sum()
n_fail = len(df) - n_pass

print(f"\nDataset composition:")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"  TOTAL: {len(df)}  |  Ratio PASS:FAIL = {n_pass/n_fail:.1f}:1")

# ── Encode categorical features ──────────────────────────────
le_proto = LabelEncoder()
le_site  = LabelEncoder()
df['Protocol_enc'] = le_proto.fit_transform(df['Protocol'].astype(str))
df['Site_enc']     = le_site.fit_transform(df['Site'].astype(str))

print(f"\n  Protocols: {list(le_proto.classes_)}")
print(f"  Sites:     {list(le_site.classes_)}")

FEATURE_COLS = METRIC_COLS + ['Protocol_enc', 'Site_enc']
X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))
y = df['QA_Label']


# =============================================================
# 2. MODEL DEFINITIONS
# =============================================================
print("\n" + "=" * 60)
print(" MODEL DEFINITIONS")
print("=" * 60)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
}
if HAS_XGB:
    scale = (y == 1).sum() / (y == 0).sum()
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        verbosity=0,
        eval_metric='logloss'
    )

for name in models:
    print(f"  ✓ {name}")


# =============================================================
# 3. STRATIFIED 5-FOLD CROSS VALIDATION
# =============================================================
print("\n" + "=" * 60)
print(" STRATIFIED 5-FOLD CROSS VALIDATION")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
roc_data = {}

for name, model in models.items():
    cv = cross_validate(
        model, X, y, cv=skf,
        scoring=['roc_auc', 'f1', 'precision', 'recall', 'accuracy'],
        return_train_score=False
    )
    results[name] = cv

    y_prob = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr), thresholds, y_prob)

    print(f"\n{'─'*50}")
    print(f" {name}")
    print(f"{'─'*50}")
    print(f"  AUC       : {cv['test_roc_auc'].mean():.3f} ± {cv['test_roc_auc'].std():.3f}")
    print(f"  Accuracy  : {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
    print(f"  F1        : {cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}")
    print(f"  Precision : {cv['test_precision'].mean():.3f} ± {cv['test_precision'].std():.3f}")
    print(f"  Recall    : {cv['test_recall'].mean():.3f} ± {cv['test_recall'].std():.3f}")


# =============================================================
# 4. COMPARISON TABLE
# =============================================================
print("\n" + "=" * 60)
print(" RESULTS COMPARISON TABLE")
print("=" * 60)

summary = pd.DataFrame({
    name: {
        "AUC (mean)":  f"{v['test_roc_auc'].mean():.3f}",
        "AUC (±std)":  f"±{v['test_roc_auc'].std():.3f}",
        "Accuracy":    f"{v['test_accuracy'].mean():.3f}",
        "F1 Score":    f"{v['test_f1'].mean():.3f}",
        "Precision":   f"{v['test_precision'].mean():.3f}",
        "Recall":      f"{v['test_recall'].mean():.3f}",
    }
    for name, v in results.items()
}).T
print(summary.to_string())
best = max(results, key=lambda n: results[n]['test_roc_auc'].mean())
print(f"\n Best model (AUC): {best}")


# =============================================================
# 5. FEATURE IMPORTANCE + OPTIMAL THRESHOLD
# =============================================================
print("\n" + "=" * 60)
print(" FEATURE IMPORTANCE (full dataset training)")
print("=" * 60)

importances = {}
trained_models = {}

rf_full = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                  min_samples_split=5, min_samples_leaf=2,
                                  max_features='sqrt', random_state=42, n_jobs=-1)
rf_full.fit(X, y)
importances["Random Forest"] = pd.Series(rf_full.feature_importances_,
                                          index=FEATURE_COLS).sort_values(ascending=False)
trained_models["Random Forest"] = rf_full

if HAS_XGB:
    xgb_full = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=scale, random_state=42,
                              verbosity=0, eval_metric='logloss')
    xgb_full.fit(X, y)
    importances["XGBoost"] = pd.Series(xgb_full.feature_importances_,
                                        index=FEATURE_COLS).sort_values(ascending=False)
    trained_models["XGBoost"] = xgb_full

for mname, imp in importances.items():
    print(f"\n{mname} — Top 8:")
    for i, (feat, val) in enumerate(imp.head(8).items(), 1):
        print(f"  {i}. {feat:<25} {val:.4f}")

# Optimal threshold via Youden's J
print("\n" + "=" * 60)
print(" OPTIMAL THRESHOLD (Youden's J Index)")
print("=" * 60)
thresholds_optimal = {}
for name, (fpr, tpr, roc_auc_val, thresholds, y_prob) in roc_data.items():
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_thresh = thresholds[best_idx]
    thresholds_optimal[name] = best_thresh
    y_pred = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y, y_pred)
    sens = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    print(f"\n{name}:")
    print(f"  Optimal threshold : {best_thresh:.3f}")
    print(f"  Sensitivity       : {sens:.3f}  (correctly identified FAIL plans)")
    print(f"  Specificity       : {spec:.3f}  (correctly identified PASS plans)")
    print(f"  Confusion matrix:\n{cm}")


# =============================================================
# 6. SAVE MODELS
# =============================================================
print("\n" + "=" * 60)
print(" SAVING MODELS")
print("=" * 60)

for mname, model in trained_models.items():
    fname = f"model1_{'RF' if 'Forest' in mname else 'XGB'}.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': FEATURE_COLS,
        'metric_cols': METRIC_COLS,
        'le_proto': le_proto,
        'le_site': le_site,
        'threshold': thresholds_optimal.get(mname, 0.5)
    }, fname)
    print(f"  ✓ Saved: {fname}")


# =============================================================
# 7. PREDICTION FUNCTION
# =============================================================
def predict_qa(metrics_dict, protocol, site, model_file='model1_RF.pkl'):
    """
    Predict QA outcome for a new plan.

    Parameters:
    -----------
    metrics_dict : dict — complexity metrics from all_metrics.py output
    protocol     : str  — 'dosisoft' | '3mm3' | '3mm90' | '3mm2'
    site         : str  — 'Prostate' | 'Gynecological' | 'Pelvic'
    model_file   : str  — path to saved model .pkl

    Returns:
    --------
    dict with prediction, probability, and recommendation
    """
    saved = joblib.load(model_file)
    model     = saved['model']
    le_proto  = saved['le_proto']
    le_site   = saved['le_site']
    threshold = saved['threshold']
    feat_cols = saved['feature_cols']
    met_cols  = saved['metric_cols']

    proto_enc = le_proto.transform([protocol])[0] if protocol in le_proto.classes_ else 0
    site_enc  = le_site.transform([site])[0]      if site  in le_site.classes_  else 0

    row = [metrics_dict.get(m, np.nan) for m in met_cols] + [proto_enc, site_enc]
    X_new = pd.DataFrame([row], columns=feat_cols)
    X_new = X_new.fillna(X_new.median(numeric_only=True))

    prob_fail = model.predict_proba(X_new)[0][0]
    prob_pass = model.predict_proba(X_new)[0][1]
    prediction = 'PASS' if prob_pass >= threshold else 'FAIL'

    return {
        'prediction': prediction,
        'prob_pass': round(prob_pass*100, 1),
        'prob_fail': round(prob_fail*100, 1),
        'recommendation': (
            f"✅ QA expected to PASS with {protocol} (confidence: {prob_pass*100:.1f}%)"
            if prediction == 'PASS'
            else f"⚠️  QA likely to FAIL with {protocol} (risk: {prob_fail*100:.1f}%) — consider re-planning or alternative protocol"
        )
    }

print("\n✅ predict_qa() function defined")


# =============================================================
# 8. PLOTS
# =============================================================
print("\n" + "=" * 60)
print(" GENERATING PLOTS")
print("=" * 60)

colors = ["#55A868", "#C44E52", "#4C72B0"]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Model 1 — QA Pass/Fail Prediction\n(N={len(df)}: {n_pass} PASS + {n_fail} FAIL)",
             fontsize=14, fontweight='bold')

# ROC curves
ax1 = axes[0, 0]
for (name, (fpr, tpr, roc_auc_val, _, _)), color in zip(roc_data.items(), colors):
    ax1.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={roc_auc_val:.3f})")
ax1.plot([0,1],[0,1],'k--',lw=1,label='Random classifier (AUC=0.5)')
ax1.set_xlabel("1 - Specificity (False Positive Rate)")
ax1.set_ylabel("Sensitivity (True Positive Rate)")
ax1.set_title("ROC Curves — QA Pass/Fail")
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# AUC boxplot
ax2 = axes[0, 1]
auc_data = [results[n]['test_roc_auc'] for n in results]
bp = ax2.boxplot(auc_data, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors[:len(results)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax2.set_xticks(range(1, len(results)+1))
ax2.set_xticklabels(list(results.keys()), rotation=15, ha='right')
ax2.set_ylabel("AUC")
ax2.set_title("AUC per Fold (5-CV)")
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 1)

# Feature importance
ax3 = axes[1, 0]
top_feats = importances["Random Forest"].head(10)
bars = ax3.barh(top_feats.index[::-1], top_feats.values[::-1],
                color="#55A868", alpha=0.8, edgecolor='white')
ax3.set_xlabel("Feature Importance")
ax3.set_title("Top 10 Features — Random Forest Classifier")
for bar, val in zip(bars, top_feats.values[::-1]):
    ax3.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

# Class distribution
ax4 = axes[1, 1]
n_fail_proto = (df_gpr_clean['QA_Label'] == 0).sum()
n_fail_all = len(df_fail_clean)
labels = [f'PASS\n({n_pass})', f'FAIL - protocol\n({n_fail_proto})', f'FAIL - all\n({n_fail_all})']
sizes  = [n_pass, n_fail_proto, n_fail_all]
clrs   = ['#55A868', '#FF8C00', '#C44E52']
bars4 = ax4.bar(labels, sizes, color=clrs, alpha=0.8, edgecolor='white')
for bar, val in zip(bars4, sizes):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             str(val), ha='center', fontweight='bold')
ax4.set_ylabel("Number of plans")
ax4.set_title(f"Dataset Composition\n(Total N={len(df)})")
ax4.set_ylim(0, max(sizes) * 1.2)

plt.tight_layout()
plt.savefig("model1_classification.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved as 'model1_classification.png'")
print("✅ Model 1 complete!")