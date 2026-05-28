def detect_dataset_context(df):
    """
    Lightweight domain/context detector.
    This helps avoid risky cleaning decisions.
    """
    cols = " ".join([str(c).lower() for c in df.columns])

    if any(x in cols for x in ["debit", "credit", "memo", "vendor", "account", "transaction"]):
        return "financial_transaction"

    if any(x in cols for x in ["order", "sales", "revenue", "customer", "product"]):
        return "sales_or_ecommerce"

    if any(x in cols for x in ["employee", "salary", "department", "hire"]):
        return "hr"

    if any(x in cols for x in ["stock", "inventory", "warehouse", "sku"]):
        return "inventory"

    return "general_business"


def decide_cleaning_actions(profile, dataset_context):
    """
    Converts profiler findings into safe/review decisions.
    Enterprise rule: safe fixes are automatic, risky fixes are flagged.
    """
    decisions = []

    if profile["fully_empty_rows"] > 0:
        decisions.append({
            "issue": "fully_empty_rows",
            "action": "auto_remove",
            "risk": "low",
            "confidence": 0.99,
            "reason": "Rows with no values contain no usable information."
        })

    if profile["fully_empty_columns"]:
        decisions.append({
            "issue": "fully_empty_columns",
            "action": "auto_remove",
            "risk": "low",
            "confidence": 0.99,
            "reason": "Columns with no values contain no usable information."
        })

    if profile["unnamed_columns"]:
        decisions.append({
            "issue": "unnamed_columns",
            "action": "preserve_and_rename",
            "risk": "medium",
            "confidence": 0.80,
            "reason": "Unnamed columns may contain important exported report values."
        })

    if profile["duplicate_rows"] > 0:
        if dataset_context == "financial_transaction":
            decisions.append({
                "issue": "duplicate_rows",
                "action": "flag_review_not_remove",
                "risk": "high",
                "confidence": 0.90,
                "reason": "Financial duplicate-looking rows may be valid repeated transactions."
            })
        else:
            decisions.append({
                "issue": "duplicate_rows",
                "action": "auto_remove_exact_duplicates",
                "risk": "medium",
                "confidence": 0.85,
                "reason": "Exact duplicate rows usually add noise in non-financial datasets."
            })

    if profile["possible_numeric_columns"]:
        decisions.append({
            "issue": "numeric_text_columns",
            "action": "auto_convert_when_confident",
            "risk": "low",
            "confidence": 0.90,
            "reason": "Columns with strong numeric pattern can be safely converted."
        })

    if profile["possible_date_columns"]:
        decisions.append({
            "issue": "date_text_columns",
            "action": "auto_convert_when_confident",
            "risk": "low",
            "confidence": 0.88,
            "reason": "Columns with strong date pattern can be safely converted."
        })

    if profile["high_missing_columns"]:
        decisions.append({
            "issue": "high_missing_columns",
            "action": "flag_review_not_drop",
            "risk": "medium",
            "confidence": 0.82,
            "reason": "High-missing columns may still contain important business signals."
        })

    return decisions
