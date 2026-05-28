import pandas as pd

def classify_missing_severity(missing_percent):
    if missing_percent == 0:
        return "none"
    if missing_percent < 5:
        return "low"
    if missing_percent < 25:
        return "medium"
    if missing_percent < 60:
        return "high"
    return "critical"


def classify_missing_type(df, column):
    col = str(column).lower()
    missing_percent = df[column].isna().mean() * 100

    if missing_percent == 100:
        return "structural_empty", "suspicious", "Column is completely empty."

    if col in ["debit", "credit"]:
        if "debit" in df.columns and "credit" in df.columns:
            both_missing = df["debit"].isna() & df["credit"].isna()
            if both_missing.sum() == 0:
                return "business_valid_sparse_pair", "expected", "Debit and credit are often mutually exclusive."

    if col in ["memo", "description", "note"]:
        return "optional_text_field", "expected", "Memo/description is often optional."

    if col in ["clr", "cleared_status"]:
        return "optional_status_field", "expected", "Cleared status can be blank."

    if "unknown_column" in col:
        return "preserved_export_structure", "review", "Unnamed column preserved because it has some values."

    if missing_percent > 60:
        return "high_missing_signal", "suspicious", "Very high missing values."

    return "ordinary_missing", "review", "Needs business review."


def build_missing_value_intelligence(df):
    records = []

    for col in df.columns:
        missing_mask = df[col].isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        missing_percent = round(float(missing_mask.mean() * 100), 2)
        missing_rows = df[missing_mask].index.tolist()

        missing_type, expected_or_suspicious, reason = classify_missing_type(df, col)

        records.append({
            "column": str(col),
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "severity": classify_missing_severity(missing_percent),
            "missing_type": missing_type,
            "expected_or_suspicious": expected_or_suspicious,
            "reason": reason,
            "sample_missing_rows": missing_rows[:20]
        })

    return pd.DataFrame(records)


def profile_dataset(df: pd.DataFrame):
    """
    Finds dataset faults before/after cleaning.
    This function does not change data.
    """
    issues = []

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    fully_empty_rows = int(df.isna().all(axis=1).sum())

    fully_empty_columns = [
        str(col) for col in df.columns if df[col].isna().all()
    ]

    unnamed_columns = [
        str(col) for col in df.columns
        if str(col).strip() == "" or "unnamed" in str(col).lower()
    ]

    duplicate_columns = [
        str(col) for col in df.columns[df.columns.duplicated()]
    ]

    high_missing_columns = [
        str(col) for col in df.columns
        if df[col].isna().mean() > 0.50
    ]

    possible_numeric_columns = []
    possible_date_columns = []
    mixed_type_columns = []

    for col in df.columns:
        series = df[col]

        cleaned_numeric = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
        )

        numeric_ratio = pd.to_numeric(cleaned_numeric, errors="coerce").notna().mean()
        date_ratio = pd.to_datetime(series, errors="coerce").notna().mean()

        if numeric_ratio >= 0.85 and df[col].dtype == "object":
            possible_numeric_columns.append(str(col))

        if date_ratio >= 0.70 and df[col].dtype == "object":
            possible_date_columns.append(str(col))

        if 0.30 < numeric_ratio < 0.85 and df[col].dtype == "object":
            mixed_type_columns.append(str(col))

    if total_missing > 0:
        issues.append(f"{total_missing} missing values found.")

    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate rows found.")

    if fully_empty_rows > 0:
        issues.append(f"{fully_empty_rows} fully empty rows found.")

    if fully_empty_columns:
        issues.append(f"{len(fully_empty_columns)} fully empty columns found: {fully_empty_columns}")

    if unnamed_columns:
        issues.append(f"{len(unnamed_columns)} unnamed columns found: {unnamed_columns}")

    if duplicate_columns:
        issues.append(f"{len(duplicate_columns)} duplicate column names found: {duplicate_columns}")

    if high_missing_columns:
        issues.append(f"{len(high_missing_columns)} columns have more than 50% missing values: {high_missing_columns}")

    if possible_numeric_columns:
        issues.append(f"{len(possible_numeric_columns)} text columns look numeric: {possible_numeric_columns}")

    if possible_date_columns:
        issues.append(f"{len(possible_date_columns)} text columns look like dates: {possible_date_columns}")

    if mixed_type_columns:
        issues.append(f"{len(mixed_type_columns)} columns may have mixed numeric/text values: {mixed_type_columns}")

    missing_intelligence = build_missing_value_intelligence(df)

    column_report = pd.DataFrame({
        "column": [str(col) for col in df.columns],
        "dtype": [str(df[col].dtype) for col in df.columns],
        "missing_count": [int(df[col].isna().sum()) for col in df.columns],
        "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
        "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        "sample_values": [
            ", ".join(df[col].dropna().astype(str).head(3).tolist())
            for col in df.columns
        ]
    })

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "fully_empty_rows": fully_empty_rows,
        "fully_empty_columns": fully_empty_columns,
        "unnamed_columns": unnamed_columns,
        "duplicate_columns": duplicate_columns,
        "high_missing_columns": high_missing_columns,
        "possible_numeric_columns": possible_numeric_columns,
        "possible_date_columns": possible_date_columns,
        "mixed_type_columns": mixed_type_columns,
        "issues_found": issues
    }

    return profile, column_report, missing_intelligence
