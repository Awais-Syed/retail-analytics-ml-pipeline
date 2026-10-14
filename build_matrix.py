
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT = ROOT / "outputs"
PRIVATE_OUTPUT = ROOT / "data" / "processed"

EXCLUDED_CODES = {
    "POST", "POSTAGE", "M", "D", "DOT",
    "BANK CHARGES", "C2", "ADJUST",
    "ADJUST2", "CRUK"
}


def build_matrix():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {RAW_FILE}")

    # Load the two years of transaction records
    sheets = pd.read_excel(RAW_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    # Normalize alternative column names
    df = df.rename(columns={
        "InvoiceNo": "Invoice",
        "UnitPrice": "Price",
        "CustomerID": "Customer ID"
    })

    print(f"Raw transactions: {len(df):,}")

    cleaning_log = []

    def clean_step(data, keep_mask, name):
        result = data.loc[keep_mask].copy()
        cleaning_log.append({
            "Cleaning Rule": name,
            "Rows Removed": len(data) - len(result),
            "Rows Remaining": len(result)
        })
        return result

    # Apply the same ordered rules as the notebook
    df = clean_step(
        df,
        ~df["Invoice"].astype(str).str.upper().str.startswith("C"),
        "Cancelled invoices"
    )

    df = clean_step(
        df, df["Quantity"] > 0, "Non-positive quantities"
    )

    df = clean_step(
        df, df["Price"] > 0, "Non-positive prices"
    )

    df = clean_step(
        df, df["Customer ID"].notna(), "Missing Customer IDs"
    )

    codes = df["StockCode"].astype(str).str.strip().str.upper()
    df = clean_step(
        df, ~codes.isin(EXCLUDED_CODES), "Non-product StockCodes"
    )

    valid_description = (
        df["Description"].notna()
        & df["Description"].astype(str).str.strip().ne("")
    )
    df = clean_step(
        df, valid_description, "Missing product descriptions"
    )

    pd.DataFrame(cleaning_log).to_csv(
        OUTPUT / "cleaning_log.csv", index=False
    )

    # Reserve the final two months for evaluation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=2)

    train = df.loc[df["InvoiceDate"] < cutoff].copy()
    test = df.loc[df["InvoiceDate"] >= cutoff].copy()

    # Filter using training data only
    popularity = train.groupby("StockCode")["Customer ID"].nunique()
    eligible = popularity[popularity >= 10].index

    interactions = (
        train.loc[
            train["StockCode"].isin(eligible),
            ["Customer ID", "StockCode"]
        ]
        .drop_duplicates()
    )

    customer_ids = train["Customer ID"].drop_duplicates()
    customer_index = {
        value: i for i, value in enumerate(customer_ids)
    }
    product_index = {
        value: i for i, value in enumerate(eligible)
    }

    rows = interactions["Customer ID"].map(customer_index).to_numpy()
    cols = interactions["StockCode"].map(product_index).to_numpy()
    values = np.ones(len(interactions), dtype=np.float32)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(customer_index), len(product_index)),
        dtype=np.float32
    )

    # Save matrix and mapping files
    save_npz(OUTPUT / "customer_product_matrix.npz", matrix)

    with open(OUTPUT / "product_index.json", "w") as f:
        json.dump(
            {str(i): str(code) for code, i in product_index.items()},
            f, indent=2
        )

    with open(PRIVATE_OUTPUT / "customer_index.json", "w") as f:
        json.dump(
            {str(i): str(cid) for cid, i in customer_index.items()},
            f, indent=2
        )

    # Save clean transactions and split details privately
    df.to_pickle(PRIVATE_OUTPUT / "cleaned_transactions.pkl")
    train.to_pickle(PRIVATE_OUTPUT / "train.pkl")
    test.to_pickle(PRIVATE_OUTPUT / "test.pkl")

    print(f"Cleaned records: {len(df):,}")
    print(f"Training records: {len(train):,}")
    print(f"Testing records: {len(test):,}")
    print(f"Cutoff: {cutoff}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Nonzero values: {matrix.nnz}")
    print("Matrix pipeline completed successfully.")


if __name__ == "__main__":
    build_matrix()
