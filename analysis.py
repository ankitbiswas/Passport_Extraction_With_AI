import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Output directory ───────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = Path("comparison_results") / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load ground truth ──────────────────────────────────────────────
gt_path = Path("GT.xlsx")
df_gt = pd.read_excel(gt_path, dtype={"passport_number": str})

for col in ["date_of_birth", "date_of_expiry"]:
    df_gt[col] = pd.to_datetime(df_gt[col], errors="coerce").dt.strftime("%Y-%m-%d")

print(f"GT shape: {df_gt.shape}")
print(f"GT columns: {df_gt.columns.tolist()}")
print(f"passport_number dtype: {df_gt['passport_number'].dtype}")
print(f"date_of_birth dtype: {df_gt['date_of_birth'].dtype}")
print(f"date_of_expiry dtype: {df_gt['date_of_expiry'].dtype}")
print(f"issuing_country dtype: {df_gt['issuing_country'].dtype}")
print(f"issuing_authority dtype: {df_gt['issuing_authority'].dtype}")
print(df_gt.head(), "\n")

# ── Load extracted data ────────────────────────────────────────────
extracted_path = Path(r"C:\passport_extraction\excel_merged\2026-05-13_09-31-30\extracted_data.xlsx")
df_extracted = pd.read_excel(extracted_path, dtype={"passport_number": str})

for col in ["date_of_birth", "date_of_expiry"]:
    if col in df_extracted.columns:
        df_extracted[col] = pd.to_datetime(df_extracted[col], errors="coerce").dt.strftime("%Y-%m-%d")

print(f"Extracted shape: {df_extracted.shape}")
print(f"Extracted columns: {df_extracted.columns.tolist()}")
print(df_extracted.head(), "\n")

# ── Merge & compare ────────────────────────────────────────────────
comparison_df = df_gt.merge(df_extracted, on="source_file", suffixes=("_gt", "_extracted"))

# Reorder columns: GT then extracted for each field, side-by-side comparison
field_order = [
    "first_name", "last_name", "passport_number", "nationality",
    "date_of_birth", "date_of_expiry", "issuing_country", "issuing_authority"
]
ordered_cols = ["source_file"]
for field in field_order:
    ordered_cols.append(f"{field}_gt")
    ordered_cols.append(f"{field}_extracted")

# Add _compare columns
for field in field_order:
    gt_col = f"{field}_gt"
    ext_col = f"{field}_extracted"
    compare_col = f"{field}_compare"

    def classify(row):
        gt_val = row[gt_col].lower() if isinstance(row[gt_col], str) else row[gt_col]
        ext_val = row[ext_col].lower() if isinstance(row[ext_col], str) else row[ext_col]
        gt_null = pd.isna(gt_val) or str(gt_val).strip() == ""
        ext_null = pd.isna(ext_val) or str(ext_val).strip() == ""
        if not gt_null and not ext_null:
            return "TP" if str(gt_val).strip() == str(ext_val).strip() else "MISTAKE"
        if gt_null and not ext_null:
            return "FP"
        if not gt_null and ext_null:
            return "FN"
        return "TN"  # both null

    comparison_df[compare_col] = comparison_df.apply(classify, axis=1)

# Reorder: source_file, then for each field: gt, extracted, compare
ordered_cols = ["source_file"]
for field in field_order:
    ordered_cols.append(f"{field}_gt")
    ordered_cols.append(f"{field}_extracted")
    ordered_cols.append(f"{field}_compare")

comparison_df = comparison_df[ordered_cols]
comparison_df.to_excel(out_dir / "comparison_results.xlsx", index=False)

# ── Metrics per field ─────────────────────────────────────────────────
print("\n=== Field-level Metrics ===\n")

metrics_rows = []
agg_tp = agg_tn = agg_fp = agg_fn = agg_mistake = 0

for field in field_order:
    compare_col = f"{field}_compare"
    counts = comparison_df[compare_col].value_counts()

    tp_raw = counts.get("TP", 0)
    tn_raw = counts.get("TN", 0)
    fp_raw = counts.get("FP", 0)
    fn_raw = counts.get("FN", 0)
    mistake = counts.get("MISTAKE", 0)

    tp = tp_raw
    tn = tn_raw
    fp = fp_raw + mistake
    fn = fn_raw + mistake

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_rows.append({
        "field": field,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "MISTAKE": mistake,
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    })

    agg_tp += tp
    agg_tn += tn
    agg_fp += fp
    agg_fn += fn
    agg_mistake += mistake

    print(f"{field}:")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}, MISTAKE={mistake}")
    print(f"  Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print()

# Aggregate "ALL" row
agg_total = agg_tp + agg_tn + agg_fp + agg_fn
agg_accuracy = (agg_tp + agg_tn) / agg_total if agg_total > 0 else 0
agg_precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0
agg_recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0
agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0

metrics_rows.append({
    "field": "ALL",
    "TP": agg_tp, "TN": agg_tn, "FP": agg_fp, "FN": agg_fn, "MISTAKE": agg_mistake,
    "accuracy": round(agg_accuracy, 4),
    "f1": round(agg_f1, 4),
    "recall": round(agg_recall, 4),
    "precision": round(agg_precision, 4),
})

print(f"ALL:")
print(f"  TP={agg_tp}, TN={agg_tn}, FP={agg_fp}, FN={agg_fn}, MISTAKE={agg_mistake}")
print(f"  Accuracy={agg_accuracy:.4f}, Precision={agg_precision:.4f}, Recall={agg_recall:.4f}, F1={agg_f1:.4f}")

# Save metrics
metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df[["field", "TP", "TN", "FP", "FN", "MISTAKE", "accuracy", "f1", "recall", "precision"]]
metrics_df.to_excel(out_dir / "metrics.xlsx", index=False)
print(f"\nMetrics saved to {out_dir / 'metrics.xlsx'}")

