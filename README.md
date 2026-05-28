# Enterprise Data Cleaner V1

This is Version 1 of an automated data cleaning pipeline.

Flow:

1. Upload CSV or Excel
2. Profiler detects faults before cleaning
3. Decision Engine decides safe/review actions
4. Cleaner applies safe fixes
5. Validator checks cleaned data
6. Builder creates downloadable CSV, Excel, and JSON reports

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```
