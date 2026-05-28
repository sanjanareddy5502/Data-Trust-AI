import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SEMANTIC_SCHEMA = {
    "vendor_name": ["vendor", "supplier", "payee", "merchant", "company name", "name"],
    "customer_name": ["customer", "client", "buyer", "account holder", "customer name"],
    "transaction_date": ["date", "transaction date", "posting date", "invoice date", "order date"],
    "transaction_number": ["num", "number", "transaction number", "reference number", "check number"],
    "memo": ["memo", "description", "note", "details", "transaction memo"],
    "account": ["account", "bank account", "ledger account", "account name"],
    "category": ["category", "class", "department", "type", "product category"],
    "split": ["split", "account split", "category split", "distribution"],
    "debit": ["debit", "withdrawal", "money out", "expense", "charge"],
    "credit": ["credit", "deposit", "money in", "income", "payment"],
    "cleared_status": ["clr", "cleared", "reconciled", "cleared status"],
    "sales": ["sales", "revenue", "amount", "total", "total amount"],
    "quantity": ["quantity", "qty", "units", "units sold"],
    "product_name": ["product", "item", "sku", "product name", "item name"],
    "region": ["region", "state", "city", "country", "location", "territory"]
}


def load_schema_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def infer_column_meanings(df):
    model = load_schema_model()

    schema_embeddings = {
        target: model.encode(examples)
        for target, examples in SEMANTIC_SCHEMA.items()
    }

    results = []

    for col in df.columns:
        col_text = str(col).replace("_", " ").replace("-", " ").lower()

        sample_values = (
            df[col]
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )

        combined_text = col_text + " " + " ".join(sample_values[:3])
        col_embedding = model.encode([combined_text])

        best_match = None
        best_score = -1

        for target, embeddings in schema_embeddings.items():
            score = cosine_similarity(col_embedding, embeddings).max()

            if score > best_score:
                best_score = score
                best_match = target

        confidence = round(float(best_score), 3)

        if confidence >= 0.60:
            recommendation = best_match
        else:
            recommendation = "unknown"

        results.append({
            "original_column": str(col),
            "predicted_meaning": recommendation,
            "confidence": confidence,
            "sample_values": sample_values
        })

    return pd.DataFrame(results)