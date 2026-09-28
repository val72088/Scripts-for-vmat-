"""
=============================================================
Model 3 — GPR Prediction (Regression)
"Can complexity metrics predict the gamma passing rate (GPR)?"

Dataset: Per_protocol.xlsx, ' GPR' sheet
  Every row has a real GPR (%) value (both ΕΠΙΤΥΧΙΑ and ΑΠΟΤΥΧΙΑ).
  Fail QA sheet is excluded — those plans were never gamma-analyzed.

KNOWN LIMITATION: GPR is heavily left-skewed (most plans 96-100%,
long tail down to ~75%). Spearman rho is reported alongside R²/MAE/RMSE
because R² is sensitive to outliers in a skewed target.

Features: 12 complexity metrics + protocol + anatomical site
Models: Random Forest Regressor + XGBoost Regressor
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
from scipy.stats import spearmanr, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print(" XGBoost not found. Install with: pip install xgboost")
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
print(" MODEL 3 — GPR PREDICTION (REGRESSION)")
print("=" * 60)

METRIC_COLS = [
    'MCSv', 'Total_MU', 'LSV_Mean', 'AAV_Mean', 'MI_Mean',
    'Leaf_Travel_mm', 'LTMCS', 'MU_per_CP_Mean', 'MU_lt_3_Percent',
    'SAS5_Mean', 'SAS10_Mean', 'AAJA_Mean'
]

def norm_proto(p):
    if not p or str(p).strip() in ('', 'nan'): return 'Unknown'
    p = str(p).replace('\xa0', ' ').lower().strip()
    if 'dosisoft' in p: return 'dosisoft'
    if '90%' in p:      return '3mm90'
    if '3% 2mm' in p or '3%2mm' in p: return '3mm2'
    if '3% 3mm' in p or '3%3mm' in p: return '3mm3'
    return 'Unknown'

def norm_site(d):
    if not d: return 'Unknown'
    d = str(d).upper()
    if 'PROSTATE' in d:                                                return 'Prostate'
    if any(x in d for x in ['ENDOMETRIUM','CERVIX','VULVA','OVARY']):  return 'Gynecological'
    if any(x in d for x in ['BLADDER','RECTUM','ANUS']):               return 'Pelvic'
    return 'Other'

df = pd.read_excel(FILE, sheet_name=' GPR')
df['GPR'] = pd.to_numeric(df['GPR (%)'], errors='coerce')
df['Protocol'] = df['Πρωτόκολλο'].apply(norm_proto)
df['Site']     = df['Διάγνωση'].apply(norm_site)
df = df.dropna(subset=['GPR'] + METRIC_COLS)

print(f"\nDataset: N={len(df)} plans (' GPR' sheet only)")

# ── Diagnostics: how skewed is the target? ──
gpr_vals = df['GPR'].values
print("\nGPR distribution diagnostics:")
for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    print(f"  p{p:<3}: {np.percentile(gpr_vals, p):.2f}")
print(f"  Skewness       : {skew(gpr_vals):.2f}")
print(f"  N below 95%    : {(gpr_vals < 95).sum()}  /  N below 90%: {(gpr_vals < 90).sum()}")
print("  ⚠️  Target is heavily left-skewed — R² alone can be misleading.")
print("      Spearman rho is reported alongside R²/MAE/RMSE below.")

# Encode categorical features
le_proto = LabelEncoder()
le_site  = LabelEncoder()
df['Protocol_enc'] = le_proto.fit_transform(df['Protocol'].astype(str))
df['Site_enc']     = le_site.fit_transform(df['Site'].astype(str))

FEATURE_COLS = METRIC_COLS + ['Protocol_enc', 'Site_enc']
X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))
y = df['GPR']


# =============================================================
# 2. MODEL DEFINITIONS
# =============================================================
print("\n" + "=" * 60)
print(" MODEL DEFINITIONS")
print("=" * 60)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )

for name in models:
    print(f"  ✓ {name}")


# =============================================================
# 3. 5-FOLD CROSS VALIDATION
# =============================================================
print("\n" + "=" * 60)
print(" 5-FOLD CROSS VALIDATION")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
oof_preds = {}

for name, model in models.items():
    cv = cross_validate(
        model, X, y, cv=kf,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    results[name] = cv

    y_pred = cross_val_predict(model, X, y, cv=kf)
    oof_preds[name] = y_pred
    rho, pval = spearmanr(y, y_pred)

    print(f"\n{'─'*50}")
    print(f" {name}")
    print(f"{'─'*50}")
    print(f"  R²             : {cv['test_r2'].mean():.3f} ± {cv['test_r2'].std():.3f}")
    print(f"  MAE            : {-cv['test_neg_mean_absolute_error'].mean():.3f} ± {cv['test_neg_mean_absolute_error'].std():.3f}")
    print(f"  RMSE           : {-cv['test_neg_root_mean_squared_error'].mean():.3f} ± {cv['test_neg_root_mean_squared_error'].std():.3f}")
    print(f"  Spearman rho   : {rho:.3f} (p={pval:.4f})")


# =============================================================
# 4. COMPARISON TABLE
# =============================================================
print("\n" + "=" * 60)
print(" RESULTS COMPARISON TABLE")
print("=" * 60)

summary_rows = {}
for name, v in results.items():
    rho, _ = spearmanr(y, oof_preds[name])
    summary_rows[name] = {
        "R² (mean)":  f"{v['test_r2'].mean():.3f}",
        "R² (±std)":  f"±{v['test_r2'].std():.3f}",
        "MAE":        f"{-v['test_neg_mean_absolute_error'].mean():.3f}",
        "RMSE":       f"{-v['test_neg_root_mean_squared_error'].mean():.3f}",
        "Spearman ρ": f"{rho:.3f}",
    }
summary = pd.DataFrame(summary_rows).T
print(summary.to_string())
best = max(results, key=lambda n: results[n]['test_r2'].mean())
print(f"\n Best model (R²): {best}")


# =============================================================
# 5. FEATURE IMPORTANCE + SAVE MODELS
# =============================================================
print("\n" + "=" * 60)
print(" FEATURE IMPORTANCE")
print("=" * 60)

importances = {}
trained_models = {}

rf_full = RandomForestRegressor(n_estimators=300, min_samples_split=5, min_samples_leaf=2,
                                 max_features='sqrt', random_state=42, n_jobs=-1)
rf_full.fit(X, y)
importances["Random Forest"] = pd.Series(rf_full.feature_importances_,
                                          index=FEATURE_COLS).sort_values(ascending=False)
trained_models["Random Forest"] = rf_full

if HAS_XGB:
    xgb_full = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             objective='reg:squarederror', random_state=42, verbosity=0)
    xgb_full.fit(X, y)
    importances["XGBoost"] = pd.Series(xgb_full.feature_importances_,
                                        index=FEATURE_COLS).sort_values(ascending=False)
    trained_models["XGBoost"] = xgb_full

for mname, imp in importances.items():
    print(f"\n{mname} — Top 8:")
    for i, (feat, val) in enumerate(imp.head(8).items(), 1):
        print(f"  {i}. {feat:<25} {val:.4f}")

for mname, model in trained_models.items():
    fname = f"model3_{'RF' if 'Forest' in mname else 'XGB'}.pkl"
    joblib.dump({
        'model': model,
        'feature_cols': FEATURE_COLS,
        'metric_cols': METRIC_COLS,
        'le_proto': le_proto,
        'le_site': le_site
    }, fname)
    print(f"\n  ✓ Saved: {fname}")


# =============================================================
# 6. PREDICTION FUNCTION
# =============================================================
def predict_gpr(metrics_dict, protocol, site, model_file='model3_RF.pkl'):
    """
    Predict expected GPR (%) for a new plan.

    Parameters:
    -----------
    metrics_dict : dict — complexity metrics from all_metrics.py output
    protocol     : str  — 'dosisoft' | '3mm3' | '3mm90' | '3mm2'
    site         : str  — 'Prostate' | 'Gynecological' | 'Pelvic'
    model_file   : str  — path to saved model .pkl

    Returns:
    --------
    dict with predicted GPR and a plain-language flag
    """
    saved = joblib.load(model_file)
    model     = saved['model']
    le_proto  = saved['le_proto']
    le_site   = saved['le_site']
    feat_cols = saved['feature_cols']
    met_cols  = saved['metric_cols']

    proto_enc = le_proto.transform([protocol])[0] if protocol in le_proto.classes_ else 0
    site_enc  = le_site.transform([site])[0]      if site  in le_site.classes_  else 0

    row = [metrics_dict.get(m, np.nan) for m in met_cols] + [proto_enc, site_enc]
    X_new = pd.DataFrame([row], columns=feat_cols)
    X_new = X_new.fillna(X_new.median(numeric_only=True))

    pred_gpr = float(model.predict(X_new)[0])

    return {
        'predicted_gpr': round(pred_gpr, 2),
        'flag': (
            "✅ Predicted GPR comfortably above typical passing threshold."
            if pred_gpr >= 95
            else "⚠️  Predicted GPR is in the lower/borderline range — consider closer review."
        )
    }

print("\n✅ predict_gpr() function defined")


# =============================================================
# 7. PLOTS
# =============================================================
print("\n" + "=" * 60)
print(" GENERATING PLOTS")
print("=" * 60)

colors = ["#55A868", "#4C72B0"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"Model 3 — GPR Prediction (Regression)\n"
             f"(RF + XGBoost, N={len(df)}, target range {gpr_vals.min():.1f}–{gpr_vals.max():.1f}%)",
             fontsize=14, fontweight='bold')

# GPR distribution
ax1 = axes[0, 0]
ax1.hist(gpr_vals, bins=20, color="#4C72B0", alpha=0.8, edgecolor='white')
ax1.axvline(np.median(gpr_vals), color='red', linestyle='--', label=f'Median={np.median(gpr_vals):.1f}')
ax1.set_xlabel("GPR (%)")
ax1.set_ylabel("Number of plans")
ax1.set_title("GPR Distribution (left-skewed)")
ax1.legend(fontsize=8)

# Predicted vs Actual
for i, (name, color) in enumerate(zip(results.keys(), colors)):
    ax = axes[0, i+1]
    ax.scatter(y, oof_preds[name], alpha=0.5, color=color, edgecolor='white', s=40)
    lims = [min(y.min(), oof_preds[name].min()), max(y.max(), oof_preds[name].max())]
    ax.plot(lims, lims, 'k--', lw=1, label='Perfect prediction')
    rho, _ = spearmanr(y, oof_preds[name])
    ax.set_xlabel("Actual GPR (%)")
    ax.set_ylabel("Predicted GPR (%)")
    ax.set_title(f"{name}\nR²={results[name]['test_r2'].mean():.3f}, ρ={rho:.3f}")
    ax.legend(fontsize=8)

# R² boxplot
ax4 = axes[1, 0]
r2_data = [results[n]['test_r2'] for n in results]
bp = ax4.boxplot(r2_data, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors[:len(results)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax4.set_xticks(range(1, len(results)+1))
ax4.set_xticklabels(list(results.keys()), rotation=15, ha='right', fontsize=9)
ax4.set_ylabel("R²")
ax4.set_title("R² per Fold (5-CV)")
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Feature importance
ax5 = axes[1, 1:]
top_feats = importances["Random Forest"].head(10)
bars = ax5.barh(top_feats.index[::-1], top_feats.values[::-1],
                color="#55A868", alpha=0.8, edgecolor='white')
ax5.set_xlabel("Feature Importance")
ax5.set_title("Top 10 Features — Random Forest Regressor")
for bar, val in zip(bars, top_feats.values[::-1]):
    ax5.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("model3_gpr_regression.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved as 'model3_gpr_regression.png'")
print("✅ Model 3 complete!")