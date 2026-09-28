"""
=============================================================================
VMAT COMPLEXITY METRICS CALCULATOR
=============================================================================
Calculates MCSv, LSV, AAV, MI, LT, LTMCS, SAS5, SAS10, APV, AAJA
from DICOM-RT Plan files (.dcm).
Outputs CSV files with summary and per-control-point metrics.

Metrics calculated:
  BASIC: MCSv, LSV, AAV
  FLUENCE: MI (Modulation Index)
  DELIVERABILITY: LT (Leaf Travel), LTMCS, MU/CP, %MU<3
  ACCURACY: SAS5, SAS10, APV, AAJA
=============================================================================
"""

# ================================================================
#  REUSABLE SCRIPT — CAN BE ADAPTED FOR OTHER DATASETS
# ================================================================
# This script extracts VMAT complexity metrics from DICOM-RT Plan files.
# It can be used with other DICOM-RT datasets by changing the file
# paths below (FOLDER_PATH, OUTPUT_FOLDER).
#
# For questions or issues, please refer to the thesis documentation.
# ================================================================

import pydicom
import numpy as np
import pandas as pd
import os
import glob
from typing import Dict, Any
from datetime import datetime

# ============================================================
#  USER SETTINGS — CHANGE THESE PATHS FOR YOUR SYSTEM
# ============================================================

# Folder containing your DICOM-RT Plan files (.dcm)
FOLDER_PATH = r"C:\Users\Admin\PycharmProjects\PythonProject\TP"

# Folder where CSV result files will be saved
OUTPUT_FOLDER = r"C:\Users\Admin\PycharmProjects\PythonProject\TP\Results"

# File pattern to match (usually *.dcm)
FILE_PATTERN = "*.dcm"

# ============================================================


def detect_aperture_sign(all_mlc_positions: np.ndarray, n_leaves: int) -> int:
    """
    Detect which sign convention is used for MLC leaf positions.
    Returns: +1 if convention is (right - left), -1 if convention is (left - right)
    """
    left_all = all_mlc_positions[:, :n_leaves]
    right_all = all_mlc_positions[:, n_leaves:]

    mean_conv1 = float(np.mean(right_all - left_all))
    mean_conv2 = float(np.mean(left_all - right_all))

    return +1 if mean_conv1 >= mean_conv2 else -1


def extract_mlc_data_from_plan(rp_filepath: str) -> Dict[str, Any]:
    """
    Read a DICOM-RT Plan file and extract the data needed for complexity analysis.
    """
    plan = pydicom.dcmread(rp_filepath, force=True)

    # Find the treatment beam
    treatment_beam = None
    for beam in plan.BeamSequence:
        if beam.TreatmentDeliveryType == "TREATMENT":
            treatment_beam = beam
            break

    if treatment_beam is None:
        raise ValueError("No treatment beam found in the DICOM file")

    cp_sequence = treatment_beam.ControlPointSequence
    num_control_points = len(cp_sequence)

    if num_control_points == 0:
        raise ValueError("No control points found in the treatment beam")

    cumulative_mu = np.zeros(num_control_points, dtype=np.float64)
    gantry_angles = np.zeros(num_control_points, dtype=np.float64)
    mlc_positions_list = []

    for i, cp in enumerate(cp_sequence):

        if hasattr(cp, 'CumulativeMetersetWeight'):
            cumulative_mu[i] = float(cp.CumulativeMetersetWeight)

        # Gantry angle correction for DICOM decimal units
        if hasattr(cp, 'GantryAngle'):
            angle = float(cp.GantryAngle)
            if angle > 36000:
                angle = angle / 10000.0
            elif angle > 3600:
                angle = angle / 1000.0
            elif angle > 360:
                angle = angle / 100.0
            gantry_angles[i] = angle

        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
            for device in cp.BeamLimitingDevicePositionSequence:
                if hasattr(device, 'RTBeamLimitingDeviceType'):
                    if device.RTBeamLimitingDeviceType == 'MLCX':
                        if hasattr(device, 'LeafJawPositions'):
                            positions = np.array(device.LeafJawPositions, dtype=np.float64)
                            mlc_positions_list.append(positions)
                            break

    if len(mlc_positions_list) == 0:
        raise ValueError("No MLC position data found in the DICOM file")

    mlc_positions_array = np.array(mlc_positions_list, dtype=np.float64)

    # Determine total MU
    real_total_mu = 0.0

    if hasattr(treatment_beam, 'BeamMeterset'):
        real_total_mu = float(treatment_beam.BeamMeterset)
    elif hasattr(plan, 'FractionGroupSequence'):
        for fraction_group in plan.FractionGroupSequence:
            if hasattr(fraction_group, 'ReferencedBeamSequence'):
                for ref_beam in fraction_group.ReferencedBeamSequence:
                    if hasattr(ref_beam, 'BeamMeterset'):
                        real_total_mu = float(ref_beam.BeamMeterset)
                        break
            if real_total_mu > 0:
                break
    elif cumulative_mu[-1] > 1.0:
        real_total_mu = cumulative_mu[-1]

    if real_total_mu == 0:
        if 0 < cumulative_mu[-1] <= 1.0:
            real_total_mu = 100.0
        else:
            real_total_mu = cumulative_mu[-1] if cumulative_mu[-1] > 0 else 100.0

    # Calculate delta MU — VECTORIZED
    is_normalized = (0 < cumulative_mu[-1] <= 1.0)
    weights = np.diff(cumulative_mu, prepend=0.0)

    if is_normalized:
        delta_mu = weights * real_total_mu
    else:
        delta_mu = weights

    normalized_mu = delta_mu / real_total_mu

    n_leaves = mlc_positions_array.shape[1] // 2 if mlc_positions_array.shape[0] > 0 else 0

    if n_leaves == 0:
        raise ValueError("Could not determine number of MLC leaves from the data")

    aperture_sign = detect_aperture_sign(mlc_positions_array, n_leaves)

    return {
        'mlc_positions': mlc_positions_array,
        'cumulative_mu': cumulative_mu,
        'delta_mu': delta_mu,
        'normalized_mu': normalized_mu,
        'total_mu': real_total_mu,
        'gantry_angles': gantry_angles,
        'n_control_points': num_control_points,
        'n_leaves': n_leaves,
        'plan_name': os.path.basename(rp_filepath),
        'aperture_sign': aperture_sign,
        'plan_object': plan
    }


def calculate_global_pos_max(all_mlc_positions: np.ndarray, n_leaves: int) -> float:
    """
    Calculate pos_max: the average range of motion of MLC leaves across the arc.
    Used to normalize the LSV score.
    """
    left_banks_all = all_mlc_positions[:, :n_leaves]
    right_banks_all = all_mlc_positions[:, n_leaves:]

    left_ranges = np.max(left_banks_all, axis=0) - np.min(left_banks_all, axis=0)
    right_ranges = np.max(right_banks_all, axis=0) - np.min(right_banks_all, axis=0)

    pos_max = (float(np.mean(left_ranges)) + float(np.mean(right_ranges))) / 2.0

    if pos_max <= 0:
        pos_max = 1.0

    return pos_max


def _calculate_bank_lsv(bank_positions: np.ndarray, pos_max: float) -> float:
    """
    Calculate LSV for a single bank of leaves at one control point.
    LSV = 1.0 → simple (uniform aperture)
    LSV → 0.0 → complex (highly irregular aperture)
    """
    n_leaves_bank = len(bank_positions)

    if n_leaves_bank <= 1 or pos_max <= 0:
        return 1.0

    total_variation = 0.0

    for n in range(n_leaves_bank - 1):
        diff = abs(bank_positions[n] - bank_positions[n + 1])
        total_variation += (pos_max - min(diff, pos_max))

    lsv_bank = total_variation / ((n_leaves_bank - 1) * pos_max)

    return max(0.0, min(1.0, lsv_bank))


def calculate_lsv_for_control_point(
        left_bank: np.ndarray,
        right_bank: np.ndarray,
        pos_max: float
) -> float:
    """Calculate LSV for a single control point (product of both banks)."""
    lsv_left = _calculate_bank_lsv(left_bank, pos_max)
    lsv_right = _calculate_bank_lsv(right_bank, pos_max)
    return lsv_left * lsv_right


def calculate_global_max_apertures(
        all_mlc_positions: np.ndarray,
        n_leaves: int,
        aperture_sign: int
) -> np.ndarray:
    """
    Calculate the maximum aperture width per leaf pair over the entire arc.
    Used to normalize the AAV score.
    """
    left_all = all_mlc_positions[:, :n_leaves]
    right_all = all_mlc_positions[:, n_leaves:]

    apertures_all = aperture_sign * (right_all - left_all)
    global_max_apertures = np.max(apertures_all, axis=0)
    global_max_apertures = np.maximum(global_max_apertures, 0.1)

    return global_max_apertures


def calculate_aav_for_control_point(
        mlc_positions_cp: np.ndarray,
        n_leaves: int,
        global_max_apertures: np.ndarray,
        aperture_sign: int
) -> float:
    """
    Calculate AAV for a single control point.
    AAV = 1.0 → all leaf pairs fully open → simple
    AAV = 0.0 → all leaf pairs closed → complex
    """
    left_bank = mlc_positions_cp[:n_leaves]
    right_bank = mlc_positions_cp[n_leaves:]

    current_apertures = aperture_sign * (right_bank - left_bank)
    current_apertures = np.maximum(current_apertures, 0.0)

    current_total = float(np.sum(current_apertures))
    max_total = float(np.sum(global_max_apertures))

    if max_total <= 0:
        return 1.0

    aav = current_total / max_total
    return min(1.0, aav)


def calculate_modulation_index(mlc_positions_cp: np.ndarray, n_leaves: int,
                               aperture_sign: int, threshold_ratio: float = 0.3) -> float:
    """
    Calculate Modulation Index (MI) - Fluence metric.
    MI = proportion of adjacent leaf pairs with difference > threshold.
    Higher MI = more modulation (more complex).
    """
    left_bank = mlc_positions_cp[:n_leaves]
    right_bank = mlc_positions_cp[n_leaves:]

    apertures = aperture_sign * (right_bank - left_bank)
    apertures = np.maximum(apertures, 0.0)

    if len(apertures) < 2 or np.std(apertures) < 1e-6:
        return 0.0

    threshold = threshold_ratio * np.std(apertures)
    differences = np.abs(np.diff(apertures))

    mi = np.sum(differences > threshold) / len(differences)

    return float(mi)


def calculate_leaf_travel(all_mlc_positions: np.ndarray, n_leaves: int, aperture_sign: int) -> float:
    """
    Calculate Leaf Travel (LT) - Deliverability metric.
    LT quantifies how much MLC leaves move across the entire arc (in mm).
    Higher LT = more leaf movement = potentially more complex.
    """
    n_cp = all_mlc_positions.shape[0]
    leaf_travel = np.zeros(n_leaves)

    for leaf_idx in range(n_leaves):
        left_positions = all_mlc_positions[:, leaf_idx]
        right_positions = all_mlc_positions[:, n_leaves + leaf_idx]

        apertures = aperture_sign * (right_positions - left_positions)
        apertures = np.maximum(apertures, 0.0)
        is_open = apertures > 1.0

        if np.any(is_open):
            left_travel = np.sum(np.abs(np.diff(left_positions[is_open]))) if len(left_positions[is_open]) > 1 else 0
            right_travel = np.sum(np.abs(np.diff(right_positions[is_open]))) if len(right_positions[is_open]) > 1 else 0
            leaf_travel[leaf_idx] = left_travel + right_travel

    return float(np.mean(leaf_travel))


def calculate_small_aperture_score(mlc_positions_cp: np.ndarray, n_leaves: int,
                                   aperture_sign: int, threshold_mm: float = 10.0) -> float:
    """
    Calculate Small Aperture Score (SAS) - Accuracy metric.
    SAS(x) = proportion of leaf pairs with aperture < x mm.
    Higher SAS = more small apertures = more complex.
    Returns percentage (0-100%).
    """
    left_bank = mlc_positions_cp[:n_leaves]
    right_bank = mlc_positions_cp[n_leaves:]

    apertures = aperture_sign * (right_bank - left_bank)
    apertures = np.maximum(apertures, 0.0)

    return float(np.mean(apertures < threshold_mm) * 100.0)


def calculate_aperture_perimeter_variation(mlc_positions_cp: np.ndarray, n_leaves: int,
                                           aperture_sign: int, leaf_width_mm: float = 5.0) -> float:
    """
    Calculate Aperture Perimeter Variation (APV) - Accuracy metric.
    APV measures how irregular the aperture shape is.
    Higher APV = more irregular aperture = more complex.
    """
    left_bank = mlc_positions_cp[:n_leaves]
    right_bank = mlc_positions_cp[n_leaves:]

    apertures = aperture_sign * (right_bank - left_bank)
    apertures = np.maximum(apertures, 0.0)

    vertical_edges = np.sum(np.abs(np.diff(apertures)))
    horizontal_edges = 2 * np.sum(apertures)
    perimeter = vertical_edges + horizontal_edges
    area = np.sum(apertures) * leaf_width_mm

    if area <= 0:
        return 0.0

    return float(perimeter / area)


def calculate_aaja(mlc_positions_cp: np.ndarray, n_leaves: int, aperture_sign: int) -> float:
    """
    Calculate Average Adjacent Jaw Aperture (AAJA) - Accuracy metric.
    AAJA measures the mean difference between adjacent leaf positions.
    Lower AAJA = more complex (more jagged aperture shape).
    """
    left_bank = mlc_positions_cp[:n_leaves]
    right_bank = mlc_positions_cp[n_leaves:]

    apertures = aperture_sign * (right_bank - left_bank)
    apertures = np.maximum(apertures, 0.0)

    if len(apertures) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(apertures))))


def calculate_mcsv_single_plan(rp_filepath: str) -> Dict[str, Any]:
    """
    Calculate MCSv and all additional complexity metrics for a single VMAT plan.
    """
    extracted_data = extract_mlc_data_from_plan(rp_filepath)
    mlc_positions = extracted_data['mlc_positions']
    delta_mu = extracted_data['delta_mu']
    normalized_mu = extracted_data['normalized_mu']
    total_mu = extracted_data['total_mu']
    n_cp = extracted_data['n_control_points']
    n_leaves = extracted_data['n_leaves']
    plan_name = extracted_data['plan_name']
    aperture_sign = extracted_data['aperture_sign']

    pos_max = calculate_global_pos_max(mlc_positions, n_leaves)
    global_max_apertures = calculate_global_max_apertures(
        mlc_positions, n_leaves, aperture_sign
    )

    lsv_values = np.zeros(n_cp, dtype=np.float64)
    aav_values = np.zeros(n_cp, dtype=np.float64)
    mi_values = np.zeros(n_cp, dtype=np.float64)
    sas_5mm_values = np.zeros(n_cp, dtype=np.float64)
    sas_10mm_values = np.zeros(n_cp, dtype=np.float64)
    apv_values = np.zeros(n_cp, dtype=np.float64)
    aaja_values = np.zeros(n_cp, dtype=np.float64)

    for i in range(n_cp):
        left_bank = mlc_positions[i, :n_leaves]
        right_bank = mlc_positions[i, n_leaves:]

        lsv_values[i] = calculate_lsv_for_control_point(left_bank, right_bank, pos_max)
        aav_values[i] = calculate_aav_for_control_point(
            mlc_positions[i], n_leaves, global_max_apertures, aperture_sign
        )
        mi_values[i] = calculate_modulation_index(mlc_positions[i], n_leaves, aperture_sign)
        sas_5mm_values[i] = calculate_small_aperture_score(mlc_positions[i], n_leaves, aperture_sign, threshold_mm=5.0)
        sas_10mm_values[i] = calculate_small_aperture_score(mlc_positions[i], n_leaves, aperture_sign, threshold_mm=10.0)
        apv_values[i] = calculate_aperture_perimeter_variation(mlc_positions[i], n_leaves, aperture_sign)
        aaja_values[i] = calculate_aaja(mlc_positions[i], n_leaves, aperture_sign)

    mean_lsv = (lsv_values[:-1] + lsv_values[1:]) / 2.0
    mean_aav = (aav_values[:-1] + aav_values[1:]) / 2.0
    contributions = mean_aav * mean_lsv * normalized_mu[1:]
    mcs_value = float(np.sum(contributions))

    leaf_travel = calculate_leaf_travel(mlc_positions, n_leaves, aperture_sign)

    # LTMCS is computed in post-processing by process_multiple_plans()
    ltmcs = 0.0

    return {
        'plan_name': plan_name,
        'mcs_v': mcs_value,
        'lsv_values': lsv_values,
        'aav_values': aav_values,
        'delta_mu': delta_mu,
        'normalized_mu': normalized_mu,
        'total_mu': total_mu,
        'n_control_points': n_cp,
        'pos_max': pos_max,
        'gantry_angles': extracted_data['gantry_angles'],
        'n_leaves': n_leaves,
        'contributions': contributions,
        'mi_values': mi_values,
        'mi_mean': float(np.mean(mi_values)),
        'mi_std': float(np.std(mi_values)),
        'mi_max': float(np.max(mi_values)),
        'leaf_travel': leaf_travel,
        'ltmcs': ltmcs,
        'mu_per_cp_mean': float(np.mean(delta_mu)),
        'mu_lt_3_percent': float(np.mean(delta_mu < 3) * 100),
        'sas_5mm_values': sas_5mm_values,
        'sas_5mm_mean': float(np.mean(sas_5mm_values)),
        'sas_5mm_max': float(np.max(sas_5mm_values)),
        'sas_10mm_values': sas_10mm_values,
        'sas_10mm_mean': float(np.mean(sas_10mm_values)),
        'sas_10mm_max': float(np.max(sas_10mm_values)),
        'apv_values': apv_values,
        'apv_mean': float(np.mean(apv_values)),
        'apv_std': float(np.std(apv_values)),
        'aaja_values': aaja_values,
        'aaja_mean': float(np.mean(aaja_values)),
        'aaja_std': float(np.std(aaja_values))
    }


def process_multiple_plans(folder_path: str, file_pattern: str = "*.dcm") -> Dict[str, Any]:
    """
    Process all DICOM-RT Plan files in a folder and collect their metrics.
    """
    search_pattern = os.path.join(folder_path, file_pattern)
    all_files = glob.glob(search_pattern)

    all_summaries = []
    all_control_points = []
    all_contributions = []
    failed_files = []
    skipped_files = []

    for filepath in all_files:
        filename = os.path.basename(filepath)

        try:
            result = calculate_mcsv_single_plan(filepath)

            all_summaries.append({
                'Plan_Name': result['plan_name'],
                'MCSv': result['mcs_v'],
                'Total_MU': result['total_mu'],
                'Num_Control_Points': result['n_control_points'],
                'pos_max_mm': result['pos_max'],
                'Num_Leaves': result['n_leaves'],
                'LSV_Mean': np.mean(result['lsv_values']),
                'LSV_Std': np.std(result['lsv_values']),
                'LSV_Min': np.min(result['lsv_values']),
                'LSV_Max': np.max(result['lsv_values']),
                'AAV_Mean': np.mean(result['aav_values']),
                'AAV_Std': np.std(result['aav_values']),
                'AAV_Min': np.min(result['aav_values']),
                'AAV_Max': np.max(result['aav_values']),
                'MI_Mean': result['mi_mean'],
                'MI_Std': result['mi_std'],
                'MI_Max': result['mi_max'],
                'Leaf_Travel_mm': result['leaf_travel'],
                'LTMCS': result['ltmcs'],
                'MU_per_CP_Mean': result['mu_per_cp_mean'],
                'MU_lt_3_Percent': result['mu_lt_3_percent'],
                'SAS5_Mean': result['sas_5mm_mean'],
                'SAS5_Max': result['sas_5mm_max'],
                'SAS10_Mean': result['sas_10mm_mean'],
                'SAS10_Max': result['sas_10mm_max'],
                'APV_Mean': result['apv_mean'],
                'APV_Std': result['apv_std'],
                'AAJA_Mean': result['aaja_mean'],
                'AAJA_Std': result['aaja_std']
            })

            for cp_idx in range(result['n_control_points']):
                all_control_points.append({
                    'Plan_Name': result['plan_name'],
                    'CP_Index': cp_idx,
                    'Gantry_Angle_deg': result['gantry_angles'][cp_idx],
                    'LSV': result['lsv_values'][cp_idx],
                    'AAV': result['aav_values'][cp_idx],
                    'MI': result['mi_values'][cp_idx],
                    'SAS5': result['sas_5mm_values'][cp_idx],
                    'SAS10': result['sas_10mm_values'][cp_idx],
                    'APV': result['apv_values'][cp_idx],
                    'AAJA': result['aaja_values'][cp_idx],
                    'Delta_MU': result['delta_mu'][cp_idx],
                    'Normalized_MU': result['normalized_mu'][cp_idx]
                })

            contributions_array = result['contributions']
            cumulative_mcsv = np.cumsum(contributions_array)
            for pair_idx in range(len(contributions_array)):
                all_contributions.append({
                    'Plan_Name': result['plan_name'],
                    'Pair_Index': pair_idx + 1,
                    'Contribution': contributions_array[pair_idx],
                    'Cumulative_MCSv': cumulative_mcsv[pair_idx]
                })

        except ValueError:
            skipped_files.append(filename)
        except Exception:
            failed_files.append(filename)

    # POST-PROCESSING: Recalculate LTMCS using LT_max across all plans
    # Formula: LTMCS = MCSv × (1 - LT / LT_max)  [Masi et al.]
    if all_summaries:
        lt_values = [s['Leaf_Travel_mm'] for s in all_summaries]
        lt_max = max(lt_values)
        for s in all_summaries:
            lt_norm = 1.0 - (s['Leaf_Travel_mm'] / lt_max)
            s['LTMCS'] = max(0.0, min(1.0, s['MCSv'] * lt_norm))

    return {
        'summaries': all_summaries,
        'control_points': all_control_points,
        'contributions': all_contributions,
        'failed_files': failed_files,
        'skipped_files': skipped_files
    }


def save_batch_results(results: Dict[str, Any], output_folder: str = "."):
    """
    Save all batch results to CSV files in the specified output folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results['summaries']:
        df_summary = pd.DataFrame(results['summaries'])
        summary_file = os.path.join(output_folder, f'all_plans_summary_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False, float_format='%.6f')

    if results['control_points'] and len(results['control_points']) < 50000:
        df_cp = pd.DataFrame(results['control_points'])
        cp_file = os.path.join(output_folder, f'all_plans_control_points_{timestamp}.csv')
        df_cp.to_csv(cp_file, index=False, float_format='%.6f')

    if results['contributions']:
        df_contrib = pd.DataFrame(results['contributions'])
        contrib_file = os.path.join(output_folder, f'all_plans_contributions_{timestamp}.csv')
        df_contrib.to_csv(contrib_file, index=False, float_format='%.6f')

    if results['failed_files'] or results['skipped_files']:
        error_log = []
        for f in results['skipped_files']:
            error_log.append({'File': f, 'Status': 'Skipped (not VMAT)'})
        for f in results['failed_files']:
            error_log.append({'File': f, 'Status': 'Failed'})
        df_errors = pd.DataFrame(error_log)
        error_file = os.path.join(output_folder, f'error_log_{timestamp}.csv')
        df_errors.to_csv(error_file, index=False)


def print_batch_summary(results: Dict[str, Any]):
    """
    Print a human-readable summary table and statistics to the console.
    """
    if not results['summaries']:
        return

    df_summary = pd.DataFrame(results['summaries'])

    print(f"\n{'=' * 80}")
    print("BATCH RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Number of plans: {len(df_summary)}")
    print(f"  MCSv range:      {df_summary['MCSv'].min():.4f} – {df_summary['MCSv'].max():.4f}")
    print(f"  MCSv mean:       {df_summary['MCSv'].mean():.4f} ± {df_summary['MCSv'].std():.4f}")
    print(f"  LT (mm) mean:    {df_summary['Leaf_Travel_mm'].mean():.1f} ± {df_summary['Leaf_Travel_mm'].std():.1f}")
    print(f"  LTMCS mean:      {df_summary['LTMCS'].mean():.4f} ± {df_summary['LTMCS'].std():.4f}")
    print(f"  MI mean:         {df_summary['MI_Mean'].mean():.4f} ± {df_summary['MI_Mean'].std():.4f}")
    print(f"  SAS10 (%) mean:  {df_summary['SAS10_Mean'].mean():.1f} ± {df_summary['SAS10_Mean'].std():.1f}")
    print("=" * 80)


# ============================================================
#  MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 80)
    print("VMAT COMPLEXITY METRICS CALCULATOR")
    print("=" * 80)
    print(f"Input folder:  {FOLDER_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"File pattern:  {FILE_PATTERN}")
    print("=" * 80)

    # Process all plans
    batch_results = process_multiple_plans(FOLDER_PATH, FILE_PATTERN)

    # Print summary and save results
    print_batch_summary(batch_results)
    save_batch_results(batch_results, OUTPUT_FOLDER)

    print("\n" + "=" * 80)
    print("ALL PLANS PROCESSED SUCCESSFULLY!")
    print("=" * 80)