"""
CSV to Excel Converter for VMAT Complexity Metrics
Converts CSV output files from all_metrics.py to Excel  format
for easier data inspection and analysis.

HOW TO USE:
1. Set the folder paths below (INPUT_FOLDER and OUTPUT_FOLDER)
2. Run: python csv_to_excel.py
3. Excel files will be saved in the output folder
"""

# ================================================================
#  REUSABLE SCRIPT — CAN BE ADAPTED FOR OTHER DATASETS
# ================================================================
# This script converts CSV files to Excel format.
# It can be used with any CSV files by changing the file paths below
# (INPUT_FOLDER, OUTPUT_FOLDER) and the pattern (FILE_PATTERN).
#
# For questions or issues, please refer to the thesis documentation.
# ================================================================

import pandas as pd
import os
import glob

# ============================================================
#  USER SETTINGS — CHANGE THESE PATHS FOR YOUR SYSTEM
# ============================================================

# Folder containing CSV files from all_metrics.py
INPUT_FOLDER = r"C:\Users\Admin\PycharmProjects\PythonProject\TP\Results"

# Folder where Excel files will be saved
OUTPUT_FOLDER = r"C:\Users\Admin\PycharmProjects\PythonProject\TP\Results"

# File pattern to match (usually *.csv)
FILE_PATTERN = "*.csv"

# ============================================================


if __name__ == "__main__":

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Find all CSV files matching the pattern
    search_pattern = os.path.join(INPUT_FOLDER, FILE_PATTERN)
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"⚠ No CSV files found in {INPUT_FOLDER} matching {FILE_PATTERN}")
        exit(0)

    print(f"\n📁 Found {len(csv_files)} CSV file(s) in: {INPUT_FOLDER}")
    print("=" * 60)

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Fix gantry angles (convert from DICOM decimal units if needed)
            if 'Gantry_Angle_deg' in df.columns:
                df['Gantry_Angle_deg'] = df['Gantry_Angle_deg'].apply(
                    lambda x: x / 100 if x > 3600 else x
                )

            # Save as Excel
            excel_name = filename.replace('.csv', '.xlsx')
            excel_path = os.path.join(OUTPUT_FOLDER, excel_name)
            df.to_excel(excel_path, index=False, engine='openpyxl')

            print(f"✓ {filename} → {excel_name}")
            print(f"  - {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    print("=" * 60)
    print(f"✓ All conversions complete! Output saved to: {OUTPUT_FOLDER}")