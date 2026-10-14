# Customers Like You — Retail Product Recommendation System

**Mini-Project 01 | Data Engineering and Machine Learning**

**Dataset:** UCI Online Retail II  
**Methods:** Truncated SVD, PCA, Cosine Similarity, K-Means Clustering  
**Project Type:** Individual — Data Engineer and ML Engineer

## 1. Project Overview

This project develops a product recommendation and customer segmentation system for GiftLane, a fictional UK online retailer.

The project addresses two business problems:

1. Recommend relevant products based on customers' previous purchases.
2. Identify customer groups with similar buying patterns to support targeted marketing campaigns.

The main challenge is handling a large, sparse customer-product matrix efficiently while preserving useful information for recommendations.

Truncated Singular Value Decomposition (SVD) was selected as the primary dimensionality reduction method and compared against Principal Component Analysis (PCA).

## 2. Project Objectives

The project aims to:

- Build a clean, reproducible data engineering pipeline.
- Convert transaction data into a sparse customer-product matrix.
- Reduce dimensionality using PCA and Truncated SVD.
- Generate product-to-product and customer-level recommendations.
- Evaluate Hit Rate@10 against a best-seller baseline.
- Compare model accuracy, compression, and computational performance.
- Identify interpretable customer segments without exposing customer identifiers.

## 3. Dataset

**Source:** UCI Machine Learning Repository — Online Retail II  
**Dataset link:** https://archive.ics.uci.edu/dataset/502/online+retail+ii  
**Period:** December 1, 2009 – December 9, 2011  
**Raw records:** 1,067,371

The dataset contains two years of transaction records, including invoice numbers, product codes, descriptions, quantities, invoice dates, prices, customer identifiers, and countries.

### Data Cleaning

Six sequential cleaning checks were implemented.

| Cleaning rule | Rows removed |
|---|---:|
| Cancelled invoices | 19,494 |
| Non-positive quantities | 3,457 |
| Non-positive prices | 2,750 |
| Missing Customer IDs | 236,121 |
| Non-product StockCodes | 2,888 |
| Missing product descriptions | 0 |
| **Total removed** | **264,710** |

**Final cleaned transactions:** 802,661  
**Customers after cleaning:** 5,852  
**Products after cleaning:** 4,623

The cleaning pipeline records removals at each step to avoid double-counting overlapping data-quality issues.

## 4. Train/Test Splitting

Transactions were divided chronologically, reserving the final two months for testing.

| Partition | Transactions |
|---|---:|
| Training | 684,098 |
| Testing | 118,563 |

**Cutoff:** October 9, 2011, 12:50 PM

For recommendation evaluation, only customer-product pairs representing products not previously purchased by the customer were held out.

- New held-out purchase pairs: 49,423
- Evaluable pairs in the training product catalogue: 44,727
- Customers evaluated: 1,913

Test-period transactions were excluded from the training matrix and model fitting. The reported recommendation evaluation covers customers and products represented in the training catalogue.

## 5. Customer-Product Matrix

A binary customer-product matrix was constructed using training transactions only.

A value of 1 indicates that a customer purchased a product during training, and 0 indicates no recorded purchase.

Products purchased by fewer than 10 distinct training customers were excluded.

| Matrix metric | Result |
|---|---:|
| Customers | 5,474 |
| Products | 3,688 |
| Matrix shape | 5,474 × 3,688 |
| Nonzero entries | 413,942 |
| Sparsity | 97.95% |
| Dense memory (float32) | 77.01 MiB |
| CSR sparse memory | 3.18 MiB |
| Memory reduction | 95.87% |

The CSR representation reduces storage by avoiding the explicit storage of zero values. This memory saving is separate from dimensionality reduction through SVD.

## 6. Dimensionality Reduction

Truncated SVD was selected as the primary method because it operates directly on sparse matrices without mean-centering.

Five component counts were evaluated.

| Components | Fit time (s) | Reconstruction error | Explained variance |
|---|---:|---:|---:|
| 10 | 0.06 | 0.8980 | 15.40% |
| 20 | 0.06 | 0.8758 | 19.52% |
| 50 | 0.13 | 0.8322 | 27.33% |
| 100 | 0.31 | 0.7808 | 36.05% |
| 200 | 0.89 | 0.7020 | 48.31% |

Increasing the component count improved reconstruction quality, but did not consistently improve recommendation accuracy.

**Working configuration:** Truncated SVD with 100 components.

The component-count comparison was exploratory and used the held-out evaluation data. Future work should use a separate validation period for model selection.

## 7. Recommendation System

Two recommendation functions were implemented.

**Product-to-product recommendations**

Products are represented by latent feature vectors from Truncated SVD. Cosine similarity is used to identify the 10 most similar products.

**Customer-level recommendations**

A customer's purchased products are used to calculate similarity-based recommendation scores across the product catalogue.

Previously purchased products are excluded from the final top-10 recommendations.

### Evaluation Results

Hit Rate@10 measures the proportion of evaluated customers for whom at least one held-out purchase appears among the top 10 recommendations.

| Model | Hit Rate@10 | Customers with hits |
|---|---:|---:|
| PCA (100 components) | 24.83% | 475 |
| Truncated SVD (100 components) | 25.88% | 495 |
| Best-seller baseline | **26.87%** | **514** |

Truncated SVD outperformed PCA in this recommendation setup, but did not outperform the best-seller baseline.

This suggests that product popularity remains a strong signal in the dataset. Further improvements are needed before recommending personalized SVD results for production deployment.

## 8. PCA vs Truncated SVD

| Metric | PCA | Truncated SVD |
|---|---:|---:|
| Components | 100 | 100 |
| Fit time | 0.48 s | 0.31 s |
| Explained variance | 36.17% | 36.05% |
| Hit Rate@10 | 24.83% | 25.88% |

PCA centers the data, while Truncated SVD processes the sparse matrix without centering.

The first PCA component emphasized variation associated with common bags, cake cases, and household products. The first SVD component appeared to capture overall purchasing activity involving widely purchased gift and household items.

Truncated SVD was retained because of its sparse-data compatibility, recorded fit speed, and stronger recommendation performance in this experiment.

## 9. Customer Segmentation

K-Means clustering was applied to normalized 100-dimensional SVD customer representations.

Five customer segments were identified.

| Cluster | Segment | Customers |
|---|---|---:|
| 0 | Everyday Gifts & Baking | 1,601 |
| 1 | Bags & Accessories | 947 |
| 2 | Regency Tea & Tableware | 442 |
| 3 | Home Décor & Heart Gifts | 1,337 |
| 4 | Creative Gifts & Activities | 1,147 |

Segments were interpreted using their most commonly purchased products, customer purchasing variety, and countries.

The United Kingdom was the dominant country in all five segments.

A two-dimensional SVD customer map was produced for visual interpretation. The plot displays two components, while K-Means used all 100 customer features.

**Visualization:** `outputs/customer_segments_final.png`

## 10. Reproducibility

The project uses fixed random seeds and a Python environment recorded in `requirements.txt`.

### Installation

Create and activate a Python virtual environment, then install the dependencies.

```bash
python -m pip install -r requirements.txt
```

### Dataset Setup

Download Online Retail II from the UCI Machine Learning Repository.

Extract the Excel workbook to:

`data/raw/online_retail_II.xlsx`

### Build the Matrix

Run the following from the project root:

```bash
python build_matrix.py
```

The script loads the raw data, applies cleaning rules, creates the chronological split, and generates the sparse customer-product matrix and mappings.

Expected validation results:

- Cleaned records: 802,661
- Matrix shape: (5474, 3688)
- Nonzero entries: 413942

### Notebook

Open `notebook/data_exploration.ipynb` in Jupyter or VS Code and run its analysis cells in sequence.

Raw transaction data, private customer mappings, and large processed datasets are excluded from the Git repository.

## 11. Project Limitations

- The best-seller baseline performed better than the current personalized model.
- Product popularity may influence recommendations strongly.
- Binary purchases do not capture order frequency, quantity, or recency.
- Products absent from the training catalogue cannot be recommended.
- Clustering results depend on the number of clusters and selected features.
- Selecting the number of components using test results can bias performance estimates.
- The measured fit and evaluation times represent local experiments, not a complete production nightly pipeline.

## 12. Future Improvements

Potential next steps include:

- Use separate chronological training, validation, and testing periods.
- Evaluate purchase frequency, recency, and log-scaled quantities.
- Improve recommendation scoring and combine SVD with product popularity.
- Test recommendation quality across different customer groups.
- Measure full pipeline runtime and process-level memory usage.
- Validate customer segments through marketing experiments.

## 13. Conclusion

This project demonstrates how a large retail transaction dataset can be cleaned, transformed into a sparse matrix, compressed, and used for recommendation and customer segmentation.

Sparse storage reduced the training matrix memory requirement by 95.87%. Truncated SVD generated interpretable latent representations and achieved a 25.88% Hit Rate@10.

Although this result did not exceed the 26.87% best-seller baseline, the evaluation highlights an important engineering principle: a more complex model should only be adopted when it demonstrates measurable value over a simpler alternative.

The five customer segments provide an additional foundation for product-specific marketing strategies.

## 14. Data Attribution

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Dataset license: CC BY 4.0.

GiftLane and the stakeholder scenario are fictional; the transaction data is real.