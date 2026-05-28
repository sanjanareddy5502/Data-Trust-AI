def validate_cleaned_data(raw_df, clean_df, cleaning_summary):
    """
    Validates whether cleaning damaged data or left major problems.
    """
    issues = []
    warnings = []

    raw_rows, raw_cols = raw_df.shape
    clean_rows, clean_cols = clean_df.shape

    if clean_df.empty:
        issues.append("Dataset became empty after cleaning.")

    if clean_cols == 0:
        issues.append("All columns were removed.")

    row_loss_percent = ((raw_rows - clean_rows) / raw_rows * 100) if raw_rows else 0
    column_loss_percent = ((raw_cols - clean_cols) / raw_cols * 100) if raw_cols else 0

    if row_loss_percent > 30:
        issues.append(f"High row loss: {row_loss_percent:.2f}% rows removed.")
    elif row_loss_percent > 10:
        warnings.append(f"Moderate row loss: {row_loss_percent:.2f}% rows removed.")

    if column_loss_percent > 60:
        issues.append(f"High column loss: {column_loss_percent:.2f}% columns removed.")
    elif column_loss_percent > 30:
        warnings.append(f"Moderate column loss: {column_loss_percent:.2f}% columns removed.")

    remaining_duplicates = int(clean_df.duplicated().sum())
    if remaining_duplicates > 0:
        warnings.append(f"{remaining_duplicates} duplicate rows remain after cleaning.")

    high_missing_columns = [
        str(col) for col in clean_df.columns
        if clean_df[col].isna().mean() > 0.70
    ]

    if high_missing_columns:
        warnings.append(f"Columns still having more than 70% missing values: {high_missing_columns}")

    unknown_columns = [
        str(col) for col in clean_df.columns
        if "unknown_column" in str(col)
    ]

    if unknown_columns:
        warnings.append(f"Meaningful unnamed columns preserved and may need labeling: {unknown_columns}")

    review_actions = [
        action for action in cleaning_summary.get("cleaning_log", [])
        if action.get("risk") in ["review", "high"]
    ]

    if review_actions:
        warnings.append(f"{len(review_actions)} actions need awareness/review.")

    trust_score = 100
    trust_score -= min(row_loss_percent * 0.8, 25)
    trust_score -= min(column_loss_percent * 0.7, 25)
    trust_score -= len(issues) * 20
    trust_score -= len(warnings) * 5
    trust_score = round(max(0, min(100, trust_score)), 2)

    if issues:
        status = "Needs Manual Review"
    elif trust_score >= 90:
        status = "Excellent"
    elif trust_score >= 75:
        status = "Good"
    elif trust_score >= 60:
        status = "Usable With Review"
    else:
        status = "Needs Manual Review"

    return {
        "status": status,
        "trust_score": trust_score,
        "issues": issues,
        "warnings": warnings,
        "row_loss_percent": round(float(row_loss_percent), 2),
        "column_loss_percent": round(float(column_loss_percent), 2),
        "remaining_duplicate_rows": remaining_duplicates
    }
