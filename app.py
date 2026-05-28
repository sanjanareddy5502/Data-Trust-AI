import streamlit as st
import pandas as pd

from profiler import profile_dataset
from decision_engine import detect_dataset_context, decide_cleaning_actions
from cleaner import clean_dataset
from validator import validate_cleaned_data
from builder import build_outputs

try:
    from schema_intelligence import infer_column_meanings
    SCHEMA_INTELLIGENCE_AVAILABLE = True
except Exception:
    SCHEMA_INTELLIGENCE_AVAILABLE = False

try:
    from pattern_intelligence import run_pattern_intelligence
    PATTERN_INTELLIGENCE_AVAILABLE = True
except Exception:
    PATTERN_INTELLIGENCE_AVAILABLE = False

try:
    from column_intelligence import build_column_intelligence
    COLUMN_INTELLIGENCE_AVAILABLE = True
except Exception:
    COLUMN_INTELLIGENCE_AVAILABLE = False


st.set_page_config(page_title="DataTrust AI", layout="wide")


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 17px;
            color: #666;
            margin-bottom: 25px;
        }

        .hero-card {
            padding: 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
            border: 1px solid #dce8f8;
            margin-bottom: 25px;
        }

        .section-card {
            padding: 22px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #eeeeee;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            margin-bottom: 18px;
        }

        .success-card {
            padding: 18px;
            border-radius: 14px;
            background: #f0fff4;
            border: 1px solid #b7ebc6;
        }

        .warning-card {
            padding: 18px;
            border-radius: 14px;
            background: #fff8e6;
            border: 1px solid #ffd978;
        }

        .danger-card {
            padding: 18px;
            border-radius: 14px;
            background: #fff1f0;
            border: 1px solid #ffb3ad;
        }

        .small-muted {
            font-size: 14px;
            color: #777;
        }

        div[data-testid="stMetricValue"] {
            font-size: 26px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def load_file(file):
    if file.name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            try:
                return pd.read_csv(file, encoding="latin1")
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding="cp1252")

    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    raise ValueError("Unsupported file type")


def reset_app():
    st.session_state.clear()
    st.rerun()


def set_step(step):
    st.session_state["step"] = step
    st.rerun()


def clear_downstream_from_upload():
    keys = [
        "before_profile",
        "before_column_report",
        "before_missing_intelligence",
        "schema_report",
        "schema_error",
        "column_patterns",
        "sparse_pairs",
        "pattern_error",
        "column_intelligence",
        "column_intelligence_error",
        "dataset_context",
        "decisions",
        "cleaned_df",
        "cleaning_summary",
        "after_profile",
        "after_column_report",
        "after_missing_intelligence",
        "validation_report",
        "outputs"
    ]

    for key in keys:
        st.session_state.pop(key, None)


def render_status_badge(label, active=False, done=False):
    if done:
        st.sidebar.success(label)
    elif active:
        st.sidebar.info(label)
    else:
        st.sidebar.caption(label)


if "step" not in st.session_state:
    st.session_state["step"] = 1


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">DataTrust AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Autonomous data cleaning, validation, audit logging, and trust scoring for messy business datasets.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR WORKFLOW
# ============================================================

steps = {
    1: "Upload Dataset",
    2: "Scan Data Health",
    3: "Review Cleaning Plan",
    4: "Run Cleaning Engine",
    5: "Validate & Export"
}

st.sidebar.title("Workflow Control")

if st.sidebar.button("Start Over / Change File"):
    reset_app()

selected_step = st.sidebar.radio(
    "Jump to step",
    options=list(steps.keys()),
    format_func=lambda x: f"{x}. {steps[x]}",
    index=st.session_state["step"] - 1
)

st.session_state["step"] = selected_step

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Status")

render_status_badge("1. File uploaded", st.session_state["step"] == 1, "raw_df" in st.session_state)
render_status_badge("2. Scan completed", st.session_state["step"] == 2, "before_profile" in st.session_state)
render_status_badge("3. Plan generated", st.session_state["step"] == 3, "decisions" in st.session_state)
render_status_badge("4. Cleaning completed", st.session_state["step"] == 4, "cleaned_df" in st.session_state)
render_status_badge("5. Validation completed", st.session_state["step"] == 5, "validation_report" in st.session_state)

st.sidebar.markdown("---")

if "uploaded_file_name" in st.session_state:
    st.sidebar.write("Current file:")
    st.sidebar.code(st.session_state["uploaded_file_name"])


# ============================================================
# PROGRESS
# ============================================================

st.progress((st.session_state["step"] - 1) / 4)
st.write(f"### Step {st.session_state['step']} of 5 — {steps[st.session_state['step']]}")


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

if st.session_state["step"] == 1:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Upload your dataset</h2>
            <p>Start with a CSV or Excel file. Your original file is preserved, and every cleaning action will be logged.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            raw_df = load_file(uploaded_file)

            st.session_state["raw_df"] = raw_df
            st.session_state["raw_copy"] = raw_df.copy()
            st.session_state["uploaded_file_name"] = uploaded_file.name

            clear_downstream_from_upload()

            st.success(f"File uploaded successfully: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Could not load file: {e}")

    if "raw_df" in st.session_state:
        raw_df = st.session_state["raw_df"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{raw_df.shape[0]:,}")
        c2.metric("Columns", f"{raw_df.shape[1]:,}")
        c3.metric("Missing Values", f"{int(raw_df.isna().sum().sum()):,}")
        c4.metric("Duplicate Rows", f"{int(raw_df.duplicated().sum()):,}")

        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(30), use_container_width=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Clear File"):
                reset_app()

        with col2:
            if st.button("Next: Scan Data Health", type="primary"):
                set_step(2)


# ============================================================
# STEP 2 — SCAN DATA HEALTH
# ============================================================

elif st.session_state["step"] == 2:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Data Health Scan</h2>
            <p>The system checks missing values, duplicates, column quality, schema meaning, and value patterns before making any changes.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "raw_df" not in st.session_state:
        st.warning("Please upload a file first.")
        if st.button("Go to Upload"):
            set_step(1)

    else:
        raw_df = st.session_state["raw_df"]

        if st.button("Run / Re-run Data Health Scan", type="primary"):
            with st.spinner("Scanning dataset quality..."):
                before_profile, before_column_report, before_missing_intelligence = profile_dataset(raw_df)

                st.session_state["before_profile"] = before_profile
                st.session_state["before_column_report"] = before_column_report
                st.session_state["before_missing_intelligence"] = before_missing_intelligence

                if SCHEMA_INTELLIGENCE_AVAILABLE:
                    try:
                        st.session_state["schema_report"] = infer_column_meanings(raw_df)
                    except Exception as e:
                        st.session_state["schema_error"] = str(e)

                if PATTERN_INTELLIGENCE_AVAILABLE:
                    try:
                        column_patterns, sparse_pairs = run_pattern_intelligence(raw_df)
                        st.session_state["column_patterns"] = column_patterns
                        st.session_state["sparse_pairs"] = sparse_pairs
                    except Exception as e:
                        st.session_state["pattern_error"] = str(e)

                if COLUMN_INTELLIGENCE_AVAILABLE:
                    try:
                        st.session_state["column_intelligence"] = build_column_intelligence(raw_df)
                    except Exception as e:
                        st.session_state["column_intelligence_error"] = str(e)

            for key in ["decisions", "cleaned_df", "cleaning_summary", "validation_report", "outputs"]:
                st.session_state.pop(key, None)

            st.success("Data health scan completed.")

        if "before_profile" in st.session_state:
            before_profile = st.session_state["before_profile"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{before_profile['rows']:,}")
            c2.metric("Columns", f"{before_profile['columns']:,}")
            c3.metric("Missing Values", f"{before_profile['total_missing_values']:,}")
            c4.metric("Duplicate Rows", f"{before_profile['duplicate_rows']:,}")

            st.subheader("Executive Data Health Summary")

            if before_profile["issues_found"]:
                for issue in before_profile["issues_found"]:
                    st.warning(issue)
            else:
                st.success("No major issues found.")

            st.subheader("Missing Value Intelligence")

            missing_df = st.session_state["before_missing_intelligence"]

            if missing_df.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(missing_df, use_container_width=True)

            st.subheader("Column Intelligence Summary")

            if "column_intelligence" in st.session_state:
                st.dataframe(st.session_state["column_intelligence"], use_container_width=True)
            elif "column_intelligence_error" in st.session_state:
                st.warning(st.session_state["column_intelligence_error"])
            else:
                st.info("Column Intelligence is not active.")

            with st.expander("Column-Level Fault Report"):
                st.dataframe(st.session_state["before_column_report"], use_container_width=True)

            with st.expander("Schema Intelligence"):
                if "schema_report" in st.session_state:
                    st.dataframe(st.session_state["schema_report"], use_container_width=True)
                elif "schema_error" in st.session_state:
                    st.warning(st.session_state["schema_error"])
                else:
                    st.info("Schema Intelligence is not active.")

            with st.expander("Pattern Intelligence"):
                if "column_patterns" in st.session_state:
                    st.dataframe(st.session_state["column_patterns"], use_container_width=True)

                    st.write("Sparse Numeric Pair Detection")
                    sparse_pairs = st.session_state["sparse_pairs"]

                    if sparse_pairs.empty:
                        st.info("No sparse numeric pairs detected.")
                    else:
                        st.dataframe(sparse_pairs, use_container_width=True)

                elif "pattern_error" in st.session_state:
                    st.warning(st.session_state["pattern_error"])
                else:
                    st.info("Pattern Intelligence is not active.")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Back to Upload"):
                    set_step(1)

            with col2:
                if st.button("Change File"):
                    reset_app()

            with col3:
                if st.button("Next: Review Cleaning Plan", type="primary"):
                    set_step(3)


# ============================================================
# STEP 3 — CLEANING PLAN
# ============================================================

elif st.session_state["step"] == 3:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Review Cleaning Plan</h2>
            <p>Before touching the data, the system explains what it plans to fix automatically and what it will preserve for review.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "before_profile" not in st.session_state:
        st.warning("Please run the data health scan first.")
        if st.button("Go to Scan"):
            set_step(2)

    else:
        raw_df = st.session_state["raw_df"]

        if st.button("Generate / Rebuild Cleaning Plan", type="primary"):
            dataset_context = detect_dataset_context(raw_df)

            decisions = decide_cleaning_actions(
                st.session_state["before_profile"],
                dataset_context
            )

            st.session_state["dataset_context"] = dataset_context
            st.session_state["decisions"] = decisions

            for key in ["cleaned_df", "cleaning_summary", "validation_report", "outputs"]:
                st.session_state.pop(key, None)

            st.success("Cleaning plan generated.")

        if "decisions" in st.session_state:
            st.info(f"Detected Dataset Context: {st.session_state['dataset_context']}")

            decisions = st.session_state["decisions"]

            if decisions:
                decision_df = pd.DataFrame(decisions)
                st.dataframe(decision_df, use_container_width=True)

                auto_fixes = decision_df[
                    decision_df["action"].astype(str).str.contains("auto", case=False, na=False)
                ]

                review_items = decision_df[
                    decision_df["action"].astype(str).str.contains("review|flag|preserve", case=False, na=False)
                ]

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Decisions", len(decision_df))
                c2.metric("Auto-Fix Decisions", len(auto_fixes))
                c3.metric("Review / Caution", len(review_items))

            else:
                st.success("No major cleaning actions required.")

            st.markdown(
                """
                <div class="warning-card">
                    <b>Trust rule:</b> Safe fixes are applied automatically. Risky changes are preserved, flagged, and logged.
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Back to Scan"):
                    set_step(2)

            with col2:
                if st.button("Re-scan Data"):
                    for key in [
                        "before_profile",
                        "before_column_report",
                        "before_missing_intelligence",
                        "schema_report",
                        "column_patterns",
                        "sparse_pairs",
                        "column_intelligence",
                        "decisions"
                    ]:
                        st.session_state.pop(key, None)
                    set_step(2)

            with col3:
                if st.button("Next: Run Cleaning", type="primary"):
                    set_step(4)


# ============================================================
# STEP 4 — RUN CLEANING
# ============================================================

elif st.session_state["step"] == 4:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Run Cleaning Engine</h2>
            <p>The cleaner applies the approved safe fixes and creates a full audit log of every action.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "decisions" not in st.session_state:
        st.warning("Please generate the cleaning plan first.")
        if st.button("Go to Cleaning Plan"):
            set_step(3)

    else:
        if st.button("Run / Re-run Cleaning Engine", type="primary"):
            with st.spinner("Cleaning dataset safely..."):
                cleaned_df, cleaning_summary = clean_dataset(
                    st.session_state["raw_df"],
                    st.session_state["decisions"],
                    st.session_state["dataset_context"]
                )

                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_summary"] = cleaning_summary

                for key in [
                    "after_profile",
                    "after_column_report",
                    "after_missing_intelligence",
                    "validation_report",
                    "outputs"
                ]:
                    st.session_state.pop(key, None)

            st.success("Cleaning completed.")

        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            cleaning_summary = st.session_state["cleaning_summary"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Rows", f"{cleaning_summary['original_rows']:,}")
            c2.metric("Final Rows", f"{cleaning_summary['final_rows']:,}")
            c3.metric("Rows Removed", f"{cleaning_summary['rows_removed']:,}")
            c4.metric("Actions Applied", f"{cleaning_summary['total_actions']:,}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Original Columns", f"{cleaning_summary['original_columns']:,}")
            c2.metric("Final Columns", f"{cleaning_summary['final_columns']:,}")
            c3.metric("Columns Removed", f"{cleaning_summary['columns_removed']:,}")

            with st.expander("Full Cleaning Audit Log"):
                st.json(cleaning_summary)

            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_df.head(50), use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Back to Plan"):
                    set_step(3)

            with col2:
                if st.button("Change File"):
                    reset_app()

            with col3:
                if st.button("Next: Validate & Export", type="primary"):
                    set_step(5)


# ============================================================
# STEP 5 — VALIDATE AND EXPORT
# ============================================================

elif st.session_state["step"] == 5:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Validate & Export</h2>
            <p>Final validation checks whether the cleaned dataset is trustworthy, safe, and ready for downstream use.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "cleaned_df" not in st.session_state:
        st.warning("Please run cleaning first.")
        if st.button("Go to Cleaning"):
            set_step(4)

    else:
        cleaned_df = st.session_state["cleaned_df"]

        if st.button("Run / Re-run Final Validation", type="primary"):
            with st.spinner("Validating cleaned dataset..."):
                after_profile, after_column_report, after_missing_intelligence = profile_dataset(cleaned_df)

                validation_report = validate_cleaned_data(
                    st.session_state["raw_copy"],
                    cleaned_df,
                    st.session_state["cleaning_summary"]
                )

                outputs = build_outputs(
                    clean_df=cleaned_df,
                    before_profile=st.session_state["before_profile"],
                    after_profile=after_profile,
                    decisions=st.session_state["decisions"],
                    cleaning_summary=st.session_state["cleaning_summary"],
                    validation_report=validation_report
                )

                st.session_state["after_profile"] = after_profile
                st.session_state["after_column_report"] = after_column_report
                st.session_state["after_missing_intelligence"] = after_missing_intelligence
                st.session_state["validation_report"] = validation_report
                st.session_state["outputs"] = outputs

            st.success("Validation completed.")

        if "validation_report" in st.session_state:
            validation_report = st.session_state["validation_report"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Data Trust Score", validation_report["trust_score"])
            c2.metric("Status", validation_report["status"])
            c3.metric("Remaining Duplicates", validation_report["remaining_duplicate_rows"])

            if validation_report["status"] in ["Excellent", "Good"]:
                st.markdown(
                    """
                    <div class="success-card">
                        <b>Dataset approved:</b> Cleaned dataset is ready to use.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif validation_report["status"] == "Usable With Review":
                st.markdown(
                    """
                    <div class="warning-card">
                        <b>Review recommended:</b> Dataset is usable, but some warnings remain.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="danger-card">
                        <b>Manual review needed:</b> Dataset should be reviewed before production use.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("Validation Details"):
                st.json(validation_report)

            with st.expander("After-Cleaning Missing Value Intelligence"):
                after_missing = st.session_state["after_missing_intelligence"]

                if after_missing.empty:
                    st.success("No missing values remain.")
                else:
                    st.dataframe(after_missing, use_container_width=True)

            with st.expander("After-Cleaning Column Report"):
                st.dataframe(st.session_state["after_column_report"], use_container_width=True)

            outputs = st.session_state["outputs"]

            st.subheader("Download Center")

            d1, d2, d3 = st.columns(3)

            with d1:
                st.download_button(
                    "Download Cleaned CSV",
                    outputs["csv"],
                    "cleaned_dataset.csv",
                    "text/csv"
                )

            with d2:
                st.download_button(
                    "Download Audit Report JSON",
                    outputs["json_report"],
                    "data_cleaning_audit_report.json",
                    "application/json"
                )

            with d3:
                st.download_button(
                    "Download Excel Workbook",
                    outputs["excel"],
                    "cleaned_dataset_with_audit_report.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Back to Cleaning"):
                    set_step(4)

            with col2:
                if st.button("Back to Plan"):
                    set_step(3)

            with col3:
                if st.button("Start Over With New File"):
                    reset_app()