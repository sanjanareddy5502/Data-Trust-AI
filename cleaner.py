import pandas as pd


def clean_column_name(col, fallback_name):
    col = str(col).strip()

    if col == "" or "unnamed" in col.lower():
        col = fallback_name

    col = col.lower().replace(" ", "_").replace("-", "_")
    cleaned = "".join(ch for ch in col if ch.isalnum() or ch == "_")

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")
    return cleaned or fallback_name


def make_unique_columns(columns):
    seen = {}
    final_columns = []

    for col in columns:
        if col not in seen:
            seen[col] = 0
            final_columns.append(col)
        else:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")

    return final_columns


def clean_dataset(df, decisions, dataset_context):
    """
    Applies safe cleaning decisions and logs every change.
    """
    df_clean = df.copy()
    cleaning_log = []

    original_rows = df_clean.shape[0]
    original_columns = df_clean.shape[1]

    # Fully empty rows: always safe
    empty_rows = int(df_clean.isna().all(axis=1).sum())
    if empty_rows > 0:
        df_clean = df_clean.dropna(how="all")
        cleaning_log.append({
            "action": "removed_fully_empty_rows",
            "count": empty_rows,
            "risk": "low"
        })

    # Fully empty columns: always safe
    empty_columns = [col for col in df_clean.columns if df_clean[col].isna().all()]
    if empty_columns:
        df_clean = df_clean.drop(columns=empty_columns)
        cleaning_log.append({
            "action": "removed_fully_empty_columns",
            "columns": [str(c) for c in empty_columns],
            "risk": "low"
        })

    # Preserve unnamed columns if they contain values
    renamed_columns = []
    unknown_counter = 1

    for col in df_clean.columns:
        col_text = str(col).strip()

        if col_text == "" or "unnamed" in col_text.lower():
            sample_values = df_clean[col].dropna().astype(str)

            if len(sample_values) > 0:
                new_name = f"unknown_column_{unknown_counter}"
                unknown_counter += 1
                renamed_columns.append(new_name)
                cleaning_log.append({
                    "action": "preserved_meaningful_unnamed_column",
                    "old_column": str(col),
                    "new_column": new_name,
                    "sample_values": sample_values.head(5).tolist(),
                    "risk": "review"
                })
            else:
                renamed_columns.append(f"empty_unknown_column_{unknown_counter}")
                unknown_counter += 1
        else:
            renamed_columns.append(col_text)

    df_clean.columns = renamed_columns

    # Standardize column names
    old_columns = list(df_clean.columns)
    standardized_columns = [
        clean_column_name(col, f"column_{i + 1}")
        for i, col in enumerate(df_clean.columns)
    ]
    standardized_columns = make_unique_columns(standardized_columns)
    df_clean.columns = standardized_columns

    if old_columns != standardized_columns:
        cleaning_log.append({
            "action": "standardized_column_names",
            "old_columns": [str(c) for c in old_columns],
            "new_columns": [str(c) for c in standardized_columns],
            "risk": "low"
        })

    # Trim text and standardize null-like strings
    object_columns = df_clean.select_dtypes(include=["object"]).columns

    for col in object_columns:
        before_missing = int(df_clean[col].isna().sum())

        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(
            ["", "nan", "NaN", "None", "NULL", "null", "N/A", "n/a"],
            pd.NA
        )

        after_missing = int(df_clean[col].isna().sum())

        cleaning_log.append({
            "action": "trimmed_text_and_standardized_nulls",
            "column": col,
            "missing_before": before_missing,
            "missing_after": after_missing,
            "risk": "low"
        })

    # Duplicate rows: context-aware
    duplicate_rows = int(df_clean.duplicated().sum())
    if duplicate_rows > 0:
        if dataset_context == "financial_transaction":
            cleaning_log.append({
                "action": "duplicate_rows_flagged_not_removed",
                "count": duplicate_rows,
                "risk": "high",
                "reason": "Financial duplicates may be valid transactions."
            })
        else:
            df_clean = df_clean.drop_duplicates()
            cleaning_log.append({
                "action": "removed_exact_duplicate_rows",
                "count": duplicate_rows,
                "risk": "medium"
            })

    # Convert obvious numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            numeric_candidate = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
            )

            converted = pd.to_numeric(numeric_candidate, errors="coerce")
            confidence = converted.notna().mean()

            if confidence >= 0.85:
                df_clean[col] = converted
                cleaning_log.append({
                    "action": "converted_text_to_numeric",
                    "column": col,
                    "confidence": round(float(confidence), 3),
                    "risk": "low"
                })

    # Convert obvious date columns
    for col in df_clean.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            converted_date = pd.to_datetime(df_clean[col], errors="coerce")
            confidence = converted_date.notna().mean()

            if confidence >= 0.70:
                df_clean[col] = converted_date
                cleaning_log.append({
                    "action": "converted_text_to_datetime",
                    "column": col,
                    "confidence": round(float(confidence), 3),
                    "risk": "low"
                })

    cleaning_summary = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "final_rows": int(df_clean.shape[0]),
        "final_columns": int(df_clean.shape[1]),
        "rows_removed": int(original_rows - df_clean.shape[0]),
        "columns_removed": int(original_columns - df_clean.shape[1]),
        "total_actions": len(cleaning_log),
        "cleaning_log": cleaning_log
    }

    return df_clean, cleaning_summary
