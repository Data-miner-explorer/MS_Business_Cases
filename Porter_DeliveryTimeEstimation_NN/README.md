# Porter Delivery Time Estimation — Neural Network Regression

Predicting food delivery times (in minutes) for **Porter**, India's largest intra-city logistics marketplace, using order details, restaurant category, delivery-partner availability, and estimated driving duration — so that accurate ETAs can be shown to customers at the point of ordering.



## Problem Statement

Porter operates in the ~$40B intra-city logistics market, connecting restaurants with delivery partners ("dashers") to fulfill customer orders. This project builds and compares several **regression models** — a Random Forest baseline and four Neural Network architectures — to predict `delivery_time_min`, computed as:

```
delivery_time_min = actual_delivery_time − created_at   (in minutes)
```

**Other real-world applications of this approach:** ride-hailing ETA prediction, grocery/quick-commerce delivery time estimation, courier and last-mile lead-time forecasting, dynamic dasher dispatch/staffing, and SLA monitoring for any on-demand delivery platform.



##  Dataset

- **Rows:** 175,777 unique delivery orders (`data_2.csv`)
- **Columns:** 14 raw features, including:
  - `market_id`, `store_primary_category`, `order_protocol`
  - `total_items`, `subtotal`, `num_distinct_items`, `min_item_price`, `max_item_price`
  - `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders`
  - `estimated_store_to_consumer_driving_duration`
  - `created_at`, `actual_delivery_time` (used to derive the target)
- **Target:** `delivery_time_min` (continuous, in minutes)
- No missing values or duplicate rows were found in the raw data.

---

##  Project Workflow

### 1. Data Loading & Structure Analysis
Initial inspection of shape, dtypes, descriptive statistics, and null/duplicate checks.

### 2. Preprocessing & Feature Engineering
- Parsed `created_at` / `actual_delivery_time` into datetime objects.
- Derived the target variable `delivery_time_min`.
- Extracted time-based features: `order_hour` (0–23) and `order_day_of_week` (0=Mon … 6=Sun).
- Filled remaining nulls (numeric → median, categorical → mode) — none were needed after parsing.
- Encoded `store_primary_category` (Label Encoding if non-numeric; kept as-is since already integer-coded).

### 3. Exploratory Data Analysis
- Distribution of delivery time (histogram + boxplot).
- Order volume by hour of day, day of week, order protocol, and market ID.
- Scatter plots of key numeric features vs. delivery time.
- Correlation heatmap across all numeric features.

### 4. Outlier Removal (IQR Method)
Applied the 1.5×IQR rule to the target and key numeric columns (dashers, outstanding orders, driving duration, subtotal, item prices).
- **Rows removed:** 24,438 (13.9%)
- **Final dataset size:** 151,339 rows

### 5. Train/Test Split & Scaling
- 80/20 train-test split (`random_state=42`)
- Features standardized with `StandardScaler`, **fit only on training data** to prevent leakage.

### 6. Baseline Model
- **Random Forest Regressor** (100 trees) trained as a benchmark for the neural networks.

### 7. Neural Network Models
Four architectures were built and tuned with `EarlyStopping` and `ReduceLROnPlateau`:

| Model | Architecture | Optimizer | Key Techniques |
|---|---|---|---|
| **Model 1 — Simple NN** | 2 hidden layers (64 → 32) | Adam | ReLU activation |
| **Model 2 — Deep NN** | 4 hidden layers (256→128→64→32) | Adam | BatchNorm, Dropout, ELU |
| **Model 3 — L2 + RMSProp** | 3 hidden layers (128→64→32) | RMSprop | L2 regularization, Leaky ReLU, Dropout |
| **Model 4 — SGD + Momentum** | 5 hidden layers (512→256→128→64→32) | SGD + Nesterov Momentum | BatchNorm, Dropout |

### 8. Model Comparison & Residual Analysis
All models evaluated on MSE, RMSE, MAE, and R² on the held-out test set; residuals of the best-performing model were analyzed for bias and heteroscedasticity.

---

## 📊 Results

| Model | MSE | RMSE (min) | MAE (min) | R² |
|---|---|---|---|---|
| **Model 1 — Simple NN** ⭐ | 0.1204 | **0.3470** | 0.2864 | **0.9982** |
| Model 2 — Deep NN (Dropout + BN) | 0.2173 | 0.4662 | 0.3684 | 0.9968 |
| Model 3 — L2 + RMSProp | 0.6880 | 0.8294 | 0.6231 | 0.9898 |
| Random Forest (baseline) | 2.7917 | 1.6708 | 1.2315 | 0.9584 |
| Model 4 — SGD + Momentum | 67.1417 | 8.1940 | 6.6350 | -0.0001 |

**Winner: Model 1 (Simple NN)** — the shallowest network achieved the best generalization, outperforming both the deeper networks and the Random Forest baseline. Model 4 failed to converge meaningfully under the chosen SGD learning rate/architecture combination.

### Key Observations
- `estimated_store_to_consumer_driving_duration` is consistently the strongest single predictor of delivery time.
- `total_busy_dashers` and `total_outstanding_orders` capture demand-side pressure and meaningfully affect delivery time.
- Outlier removal tightened the target's range and improved model generalization.
- BatchNorm + Dropout helped control overfitting on this tabular dataset.
- A simple, well-regularized architecture outperformed deeper/wider ones — bigger is not always better for tabular regression.

---

##  Repository Structure 

```
.
├── porter_delivery_nn_business_case.ipynb   # Main notebook
└── README.md
```

---

## Requirements

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## How to Run

1. Place `data_2.csv` in the working directory (update the path in the "Load Data" cell if needed).
2. Run the notebook top to bottom — it will:
   - Clean and engineer features
   - Train the Random Forest baseline and 4 neural network variants
   - Print evaluation metrics and generate comparison/residual plots
3. Review the final comparison table and plots in **Step 8** and **Step 9** of the notebook.



## Notebook Q&A Summary

The notebook closes with answers to conceptual questions covering:
- Pandas datetime handling (`pd.to_datetime`, `.dt.hour`, `.dt.dayofweek`)
- Datetime vs. Timedelta vs. Period
- Why and how to detect/remove outliers (IQR, Z-score, Winsorization)
- Classical ML alternatives (Gradient Boosting, Ridge/Lasso, SVR, KNN)
- Why feature scaling matters for neural networks
- Comparison of optimizers (Adam, RMSprop, SGD+Momentum) and activation functions (ReLU, ELU, Leaky ReLU)
- Why neural networks tend to scale better than tree ensembles on large datasets

See the notebook's final section for full details.
