import pandas as pd


def build_column_intelligence(df):
    rows = []

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        missing_percent = round(float(series.isna().mean() * 100), 2)
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = round(float(unique_count / max(len(series), 1)), 3)

        if pd.api.types.is_numeric_dtype(series):
            column_type = "numeric"
            insight = {
                "min": round(float(series.min()), 2) if non_null.shape[0] else None,
                "max": round(float(series.max()), 2) if non_null.shape[0] else None,
                "mean": round(float(series.mean()), 2) if non_null.shape[0] else None,
                "zero_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum())
            }

        elif pd.api.types.is_datetime64_any_dtype(series):
            column_type = "date"
            insight = {
                "min_date": str(series.min()),
                "max_date": str(series.max())
            }

        elif unique_ratio <= 0.25:
            column_type = "categorical"
            top_values = series.value_counts(dropna=True).head(5)
            insight = {
                "top_values": [
                    f"{idx} ({count})"
                    for idx, count in top_values.items()
                ]
            }

        elif unique_ratio >= 0.80:
            column_type = "high_cardinality_text_or_id"
            insight = {
                "representative_samples": non_null.astype(str).drop_duplicates().head(5).tolist()
            }

        else:
            column_type = "text"
            insight = {
                "representative_samples": non_null.astype(str).drop_duplicates().head(5).tolist()
            }

        rows.append({
            "column": str(col),
            "detected_type": column_type,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "smart_summary": insight
        })

    return pd.DataFrame(rows)