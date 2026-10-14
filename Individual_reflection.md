# Individual Reflection — Mini-Project 01

**Project:** Customers Like You — PCA or SVD  
**Role:** Data Engineer and Machine Learning Engineer

### What I Contributed

I independently worked on both the Data Engineering and Machine Learning components of this project. I started by exploring the Online Retail II dataset, identifying missing values, cancellations, invalid transactions, and non-product entries. After cleaning the data, I implemented a chronological train/test split and constructed a sparse customer-product matrix containing 5,474 customers and 3,688 products.

For the machine learning component, I implemented Truncated SVD and PCA, developed product-based and personalized recommendation functions, and evaluated their performance against a best-seller baseline. I also used K-Means clustering to identify five customer segments and created visualizations to interpret their purchasing patterns.

### What I Learned

The most important lesson was understanding that reducing data size and improving prediction accuracy are different objectives. Sparse storage reduced the matrix's memory requirement by 95.87%, but dimensionality reduction did not automatically improve recommendations.

I also learned why evaluating models against a simple baseline is essential. Although Truncated SVD achieved a Hit Rate@10 of 25.88%, the best-seller baseline performed slightly better at 26.87%. This helped me understand that a more complex model is not necessarily the better business solution.

### What I Would Improve

If I repeated this project, I would introduce a separate validation period to select hyperparameters without using the final test data. I would also experiment with purchase frequency, recency, and alternative recommendation-scoring approaches instead of relying only on binary purchases.

Finally, I would evaluate clustering quality more systematically and investigate whether combining product popularity with SVD improves recommendation performance.