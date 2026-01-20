# Sampling Techniques on an Imbalanced Credit Card Dataset

## Overview
In real-world machine learning problems, datasets are often imbalanced, meaning one class appears far more frequently than others. This assignment explores how different **sampling techniques** can be used to handle such imbalance and how these techniques influence the performance of various machine learning models.

Using a credit card fraud detection dataset, we first balance the data and then apply multiple probabilistic sampling methods. The goal is to observe how model accuracy changes depending on the sampling strategy used.

---

## Dataset Description
The dataset used is a **Credit Card Fraud Detection dataset**.

- **Target column:** `Class`
  - `0` → Non-fraudulent transaction  
  - `1` → Fraudulent transaction  

The original dataset is highly imbalanced, with fraud cases forming a very small percentage of the total transactions.

---

## Approach and Methodology

The steps followed in this assignment are outlined below:

Imbalanced Dataset  
→ Balance the dataset using sampling  
→ Apply probabilistic sampling techniques  
→ Train multiple machine learning models  
→ Compare model performance using accuracy  

This structured approach helps isolate the impact of sampling on model behavior.

---

## Balancing the Dataset

Since the original dataset was heavily skewed toward non-fraud transactions, the first step was to balance it.  
This was achieved using **random oversampling**, where samples from the minority class (fraud) were randomly duplicated until both classes had an equal number of records.

Balancing the dataset helps prevent models from becoming biased toward the majority class. However, oversampling can also introduce overfitting, especially when complex models are used. This effect is discussed later in the results section.

---

## Sampling Techniques Applied

Once the dataset was balanced, five probabilistic sampling techniques were applied to generate different training samples:

1. **Simple Random Sampling** – Randomly selects data points  
2. **Systematic Sampling** – Selects every k-th data point  
3. **Stratified Sampling** – Maintains class distribution in samples  
4. **Cluster Sampling** – Samples data from selected clusters  
5. **Bootstrap Sampling** – Samples data with replacement  

These techniques allow us to study how different ways of selecting data influence model accuracy.

---

## Machine Learning Models Used

To ensure a broad comparison, five different machine learning models were selected, each representing a different learning approach:

- **Logistic Regression (M1):** A simple linear baseline model  
- **Decision Tree (M2):** A rule-based, non-linear model  
- **Random Forest (M3):** An ensemble model known for robustness  
- **K-Nearest Neighbors (M4):** An instance-based learning algorithm  
- **Support Vector Classifier (M5):** A margin-based classifier suitable for high-dimensional data  

This diverse selection helps analyze how different types of models respond to sampling techniques.

---

## Model Evaluation Strategy

Model performance was evaluated using **accuracy** as the metric.  
To obtain reliable results, **5-fold cross-validation** was used, and the average accuracy across folds was reported.

---

## Results (Using Oversampling)

The table below shows the accuracy obtained by each model using different sampling techniques:

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|-----------|-----------|-----------|-----------|-----------|
| M1 (Logistic Regression) | 93.45 | 88.99 | 92.87 | 97.60 | 94.59 |
| M2 (Decision Tree) | 98.77 | 90.13 | 99.84 | 89.60 | 99.67 |
| M3 (Random Forest) | 100.00 | 99.74 | 100.00 | 99.40 | 100.00 |
| M4 (KNN) | 97.79 | 77.83 | 97.62 | 99.40 | 97.79 |
| M5 (SVC) | 69.62 | 62.35 | 70.74 | 99.40 | 77.07 |

---

## Graphical Analysis

Graphs were generated to visualize how model accuracy varies across different sampling techniques.  
Line plots were used to compare model performance across sampling methods, while bar charts were used to compare different models under the same sampling strategy.

From the graphs, it is clear that:
- Random Forest consistently performs very well across sampling methods  
- Cluster and Bootstrap Sampling often lead to higher accuracy  
- Systematic Sampling tends to perform worse due to sensitivity to data ordering  

---

## Overfitting Discussion

Some models achieved near-perfect or even 100% accuracy after oversampling. This is a strong indicator of potential overfitting. Since oversampling duplicates minority-class samples, complex models such as Random Forest, Decision Tree, and SVC may memorize repeated patterns instead of learning generalizable features.

While oversampling improves performance on imbalanced datasets, it should be applied cautiously in real-world scenarios.

---

## Conclusion

This assignment highlights the importance of sampling techniques when working with imbalanced datasets. Balancing the dataset significantly improves model performance, but the choice of sampling technique strongly affects the final results. Oversampling improves accuracy but can introduce overfitting, especially for powerful models.

Overall, Random Forest achieved the best performance across most sampling techniques, and Cluster and Bootstrap Sampling showed consistently strong results.

---

