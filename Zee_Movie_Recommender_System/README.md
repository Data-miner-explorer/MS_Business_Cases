# Zee Movie Recommender System

A personalized movie recommendation engine built on the classic **MovieLens 1M**-style dataset (`movies.dat`, `ratings.dat`, `users.dat`), implementing and comparing multiple collaborative filtering approaches  **item-based Pearson correlation**, **cosine similarity / KNN**, and **matrix factorization (SVD)** with latent embeddings.

## Dataset

| File | Shape | Description |
|---|---|---|
| `movies.dat` | 3,883 × 3 | `MovieID`, `Title`, `Genres` |
| `users.dat` | 6,040 × 5 | `UserID`, `Gender`, `Age`, `Occupation`, `Zip-code` |
| `ratings.dat` | 1,000,209 × 4 | `UserID`, `MovieID`, `Rating` (1–5), `Timestamp` |

The three files are merged into a single dataframe (`df`, shape `1,000,209 × 10`) keyed on `UserID` and `MovieID`. No missing values or duplicate rows were found.

## Feature Engineering

- **`Rating_Date` / `Rating_Year`** — decoded from the Unix `Timestamp`
- **`Release_Year` / `Title_clean` / `Decade`**  parsed out of the movie title (e.g. `"Toy Story (1995)"` → year `1995`)
- **`Age_Group`** and **`Occupation_Desc`**  human-readable labels mapped from the numeric MovieLens age/occupation codes
- **`Genres_list`** and one-hot **genre dummy columns**  for genre-level analysis (18 distinct genres)

## Exploratory Data Analysis

- Rating distribution, ratings by gender/age group/occupation
- Movies released per decade (dominated by the 1990s)
- Ratings by genre (Comedy and Drama lead)
- Per-movie stats: average rating and rating count (used to identify popular vs. niche titles)

## Recommendation Approaches

### 1. Data Preparation
A **user–item pivot table** (movies × users) is built after filtering to movies and users with **≥ 30 ratings each** (972,471 ratings retained, pivot shape 2,836 × 5,289), producing two versions:
- `pivot_nan`  raw values with `NaN` for missing ratings (used for Pearson correlation)
- `pivot_filled`  `NaN` imputed with `0` (used for cosine similarity, KNN, and sparse-matrix routines)

### 2. Item-Based CF — Pearson Correlation
For a target movie, computes pairwise-complete Pearson correlation against every other movie (`corrwith`), filtered by a minimum number of co-raters, and returns the top-N most correlated titles.

### 3. Item-Based CF — Cosine Similarity
- Full item–item and user–user cosine similarity matrices via `sklearn.metrics.pairwise.cosine_similarity`
- A memory-efficient **CSR (Compressed Sparse Row)** matrix version of the same approach
- A `sklearn.neighbors.NearestNeighbors` (KNN, cosine metric) implementation, cross-validated to return identical top-5 results

### 4. Matrix Factorization — SVD
Uses the **`scikit-surprise`** library's `SVD` model (SGD-trained, Funk/Netflix-Prize style) with `n_factors=4`, `n_epochs=20`, trained on an 80/20 random split of the full ratings data.

**Evaluation:**
- **RMSE (d=4): 0.8830**
- **MAPE (d=4): 26.95%**

> Note: a naive random train/test split can produce cold-start users/items unseen in training. More robust alternatives discussed in the notebook include leave-*k*-out-per-user splits, time-based splits, and K-fold cross-validation.

### 5. Latent Embeddings for Similarity
Item (`qi`) and user (`pu`) latent factor vectors from the trained SVD model are used to recompute item–item and user–user cosine similarity in a compressed embedding space (d=4), as an alternative to the raw sparse rating vectors.

A **bonus 2D (d=2) embedding visualization** colored by primary genre shows that MF embeddings form loose, overlapping genre clusters rather than clean boundaries — genre similarity emerges only indirectly, since embeddings are optimized to reconstruct ratings, not genre labels.

### 6. User-Based Recommender (Cold-Start Simulation)
Simulates a brand-new user rating a handful of movies, then:
1. Finds existing users who rated the same movies
2. Ranks them by overlap count, takes the top 100 as candidates
3. Computes Pearson similarity between the new user and each candidate
4. Takes the top 10 most similar users and computes a similarity-weighted average rating per candidate movie
5. Returns the top 10 recommended movies by weighted score

## Key Results / Questionnaire Answers

| Question | Answer |
|---|---|
| Most active age group | 25–34 (395,556 ratings) |
| Most active occupation | College/grad student |
| Majority gender of raters | Male (75.4% of ratings) |
| Decade with most movies released | 1990s |
| Movie with most ratings | *American Beauty (1999)* — 3,428 ratings |
| Top 3 movies similar to *Liar Liar (1997)* (Pearson) | *Groove (2000)*, *Picnic (1955)*, *Heavyweights (1994)* |
| CF classification | User-based vs. Item-based |
| Similarity ranges | Pearson: −1 to +1; Cosine (on non-negative ratings): 0 to +1 |
| MF evaluation | RMSE 0.8830, MAPE 26.95% |

The notebook also includes a worked example of converting a dense matrix to **CSR sparse format** (`data`, `indices`, `indptr` arrays).

## Tech Stack

- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualization
- `scipy.sparse` (CSR matrices)
- `scikit-learn` — `cosine_similarity`, `NearestNeighbors`, `TruncatedSVD`
- `scikit-surprise` — matrix factorization (`SVD`), train/test split, RMSE evaluation

## Project Structure

```
├── Zee_Recommender_System_BusinessCase.ipynb   # Main analysis notebook
├── README.md


```

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scikit-surprise
   ```
2. Place `zee-movies.dat`, `zee-users.dat`, `zee-ratings.dat` in the working directory (`/content/` in the original notebook).
3. Run the notebook top to bottom — sections are ordered: data loading → merging → EDA/feature engineering → pivot table construction → Pearson CF → cosine/KNN CF → matrix factorization → embeddings → user-based CF → questionnaire.

## Example Usage

```python
# Item-based recommendation (Pearson correlation)
recommend_pearson('Liar Liar (1997)')

# Item-based recommendation (cosine similarity / CSR)
top5_recommendations('Liar Liar (1997)')

# Item-based recommendation (KNN)
recommend_knn('Liar Liar (1997)')

# Item-based recommendation using learned MF embeddings
recommend_from_embeddings('Liar Liar (1997)')

# User-based recommendation for a new user
new_user_ratings = {
    'Toy Story (1995)': 5,
    'Liar Liar (1997)': 4,
    'Jurassic Park (1993)': 5,
    'Men in Black (1997)': 4,
    'Independence Day (ID4) (1996)': 3,
}
```
