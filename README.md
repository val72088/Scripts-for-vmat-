# VMAT analysis scripts
This repository, contains 5 python scripts for analysis on VMAT plans.


# Important Note

The scripts `model1_qa_classification.py`, `model1_22_only.py`, `model2_protocol_tolerance.py`, and `model3_gpr_regression.py` require the confidential dataset `Per_protocol.xlsx` (patient-related clinical data) and cannot be executed without it. They are provided for reference and reproducibility of the thesis results.

The scripts `all_metrics.py` and `csv_to_excel.py` are reusable and can be adapted to other DICOM-RT datasets.



# Scripts

1. `all_metrics.py` — Extracts 13 complexity metrics from DICOM-RT plan files and generates the final dataset (CSV) used for statistical analysis.
2. `csv_to_excel.py` — Converts the CSV output from `all_metrics.py` into Excel.
3. `model1_qa_classification.py` — Trains RF/XGBoost classifiers (5-fold CV) to predict pass/fail QA outcomes from complexity metrics, protocol, and site.
4. `model1_22_only.py` — Same as above, but restricted to the 2%/2mm protocol only.
5. `model2_protocol_tolerance.py` — Trains RF/XGBoost classifiers (5-fold CV) for 3-class protocol tolerance prediction.
6. `model3_gpr_regression.py` — Trains RF/XGBoost regressors (5-fold CV) to predict continuous GPR from complexity metrics.




# Requirements

 `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `openpyxl`, `joblib`, `pydicom`, `xgboost`
