import json
from io import BytesIO
import pandas as pd


def build_outputs(clean_df, before_profile, after_profile, decisions, cleaning_summary, validation_report):
    """
    Creates clean dataset and audit report outputs.
    """
    full_report = {
        "before_profile": before_profile,
        "decision_engine": decisions,
        "cleaning_summary": cleaning_summary,
        "after_profile": after_profile,
        "validation_report": validation_report
    }

    csv_output = clean_df.to_csv(index=False)
    json_output = json.dumps(full_report, indent=4, default=str)

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        clean_df.to_excel(writer, sheet_name="cleaned_data", index=False)
        pd.DataFrame(decisions).to_excel(writer, sheet_name="decisions", index=False)
        pd.DataFrame(cleaning_summary["cleaning_log"]).to_excel(writer, sheet_name="cleaning_log", index=False)
        pd.DataFrame([validation_report]).to_excel(writer, sheet_name="validation", index=False)

    excel_buffer.seek(0)

    return {
        "csv": csv_output,
        "json_report": json_output,
        "excel": excel_buffer
    }
