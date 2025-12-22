Here’s your 7‑day plan for **Week 8**, focused on **unsupervised learning**:

- **Clustering** (especially K‑means)
- **Dimensionality reduction** (especially PCA)
- Using these tools for **exploratory analysis**, visualization, and basic customer segmentation.

Assume ~1.5–2.5 hours/day. Continue: reason first, then search.

---

## Overall Week 8 Goals

By the end of this week, you should be able to:

- Explain the difference between **supervised** and **unsupervised** learning.
- Use **K‑means** clustering for simple segmentation.
- Use **PCA** to reduce dimensionality and visualize high‑dimensional data.
- Interpret clusters and principal components in plain language.
- Apply clustering + PCA to at least one **real dataset** (e.g., customer data).

---

## Day 1 – Supervised vs Unsupervised & Unsupervised Mindset

**Objectives:**
- Understand what makes unsupervised learning different.
- Learn when clustering and PCA are useful.

### 1. Notebook

Create: `week8_day1_unsupervised_intro.ipynb`.

### 2. Conceptual Notes (Markdown)

Write, in your own words:

1. **Supervised learning**:
   - You have features \(X\) and a target \(y\).
   - Goal: learn mapping \(f(X) \to y\).
   - Examples: predicting price (regression), predicting churn (classification).

2. **Unsupervised learning**:
   - You have features \(X\), **no labels**.
   - Goal: discover structure/patterns in data.
   - Examples:
     - Clustering customers into segments.
     - Reducing dimensionality for visualization.

3. **K‑means clustering (high level)**:
   - Choose k cluster centers.
   - Assign each point to the nearest center.
   - Update centers to be mean of assigned points.
   - Repeat until convergence.

4. **PCA (high level)**:
   - Finds new axes (directions) that capture **maximum variance**.
   - First principal component: direction of max variance.
   - Second component: orthogonal direction with next highest variance, etc.
   - Used for:
     - Dimensionality reduction.
     - Visualization in 2D/3D.

### 3. Pick a Dataset for Unsupervised Exploration

Choose a simple dataset this week for clustering, for example:

- Kaggle “Mall Customers” dataset (very common for clustering).
- Or any tabular dataset with customer/record features, not necessarily with labels.

Download and save as `data/mall_customers.csv` or similar.

### 4. Quick Look at the Dataset

```python
import pandas as pd

df = pd.read_csv("data/mall_customers.csv")  # adjust name
df.head()
df.info()
df.describe()
```

In Markdown:

- Which columns seem numerical and suitable for clustering?  
  (e.g., Age, Annual Income, Spending Score).
- Which columns are just IDs or non‑useful for clustering?

### 5. Thinking Challenge

Write a short brainstorm:

- If this is a customer dataset:
  - What might be **business reasons** to cluster customers?
  - What could you do with 3–5 distinct customer segments?
- What potential pitfalls could there be (e.g., sensitive attributes, overinterpreting clusters)?

Write 8–10 sentences.

---

## Day 2 – Data Preparation for Clustering & K‑means Basics

**Objectives:**
- Prepare a clean feature matrix for clustering.
- Run and understand your first K‑means model.

### 1. Notebook

Create: `week8_day2_kmeans_basics.ipynb`.

Load your chosen dataset as `df`.

### 2. Select Features to Cluster On

Pick a subset of numeric features (start small, e.g., 2–4):

Example for Mall Customers:
- `features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]`

Create:

```python
X = df[features].copy()
```

Check for missing values:

```python
X.isna().sum()
```

If any, handle simply for now (e.g., fill with median or drop missing rows). Document in Markdown.

### 3. Feature Scaling

K‑means is **distance-based**, so scale features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

In Markdown:
- Why scaling matters for K‑means (if one feature has much larger scale, it dominates distances).

### 4. Run K‑means with a Chosen k (e.g., k=3)

```python
from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_
df["cluster_k3"] = cluster_labels
```

Check basic cluster counts:

```python
df["cluster_k3"].value_counts()
```

### 5. Inspect Cluster Characteristics

Compute mean of features by cluster:

```python
df.groupby("cluster_k3")[features].mean()
```

In Markdown:
- Describe each cluster qualitatively (e.g., “Cluster 0: younger, high income, high spending”).

### 6. Thinking Challenge

- How would you **describe each cluster to a non‑technical manager**?
- If you had to name each cluster (e.g., “Budget Shoppers”, “High‑Value Customers”), what names would you choose and why?

Write 6–10 sentences.

---

## Day 3 – Visualizing Clusters in 2D (Scatter, Pairplots, Basic PCA)

**Objectives:**
- Visualize clusters.
- Get a first taste of PCA for 2D projection.

### 1. Notebook

Create: `week8_day3_cluster_visualization_pca.ipynb`.

Load `df` with `cluster_k3` and `X_scaled`.

### 2. 2D Visualization with Selected Features

If you have 2 main features (e.g., income and spending score), plot:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=X[features[1]],  # e.g., "Annual Income (k$)"
    y=X[features[2]],  # e.g., "Spending Score (1-100)"
    hue=df["cluster_k3"],
    palette="Set1"
)
plt.title("K-means Clusters (k=3)")
plt.show()
```

Interpret:
- Are clusters well separated visually?
- Any overlap?

### 3. PCA for 2D Projection (if >2 features)

Even if you have 2–3 features, practice PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Plot:

```python
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["cluster_k3"],
    palette="Set1"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters Visualized in PCA Space")
plt.show()
```

### 4. Interpret PCA Components (Lightly)

Check PCA components:

```python
import pandas as pd

components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=["PC1", "PC2"]
)
components
```

In Markdown:
- For PC1 and PC2:
  - Which features have large positive/negative weights?
  - Roughly, what does each PC represent (e.g., “overall spending power,” “age vs income trade‑off”)?

### 5. Thinking Challenge

- Compare visual separation in:
  - Raw feature space (e.g., income vs spending).
  - PCA space (PC1 vs PC2).
- Is PCA helping you see structure more clearly?
- If you had 10 or 50 features, how might PCA help before clustering or visualization?

Write 8–12 sentences.

---

## Day 4 – Choosing k: Elbow Method & Silhouette Score

**Objectives:**
- Understand that k is not known a priori.
- Use basic methods to **assess number of clusters**.

### 1. Notebook

Create: `week8_day4_choosing_k.ipynb`.

Use `X_scaled` from previous days.

### 2. Inertia & Elbow Method

Inertia = sum of squared distances of samples to nearest cluster center.

Compute for k from 1 to, say, 10:

```python
from sklearn.cluster import KMeans
import numpy as np

ks = range(1, 11)
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertias.append(km.inertia_)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(ks, inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method for k")
plt.xticks(ks)
plt.show()
```

In Markdown:
- Look for an “elbow” point where inertia stops decreasing sharply.
- Which k would you consider based on this?

### 3. Silhouette Score

Silhouette score ranges from -1 to 1 (higher = better defined clusters).

```python
from sklearn.metrics import silhouette_score

sil_scores = []

for k in range(2, 11):  # silhouette undefined for k=1
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append({"k": k, "silhouette": score})

import pandas as pd
pd.DataFrame(sil_scores)
```

Plot:

```python
plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), [d["silhouette"] for d in sil_scores], marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores vs k")
plt.xticks(range(2, 11))
plt.show()
```

### 4. Thinking Challenge

In Markdown:

- Compare elbow method and silhouette scores:
  - Do they suggest the same value of k?
  - If not, how would you decide?
- Besides these numerical methods, what **domain considerations** might influence choice of k (e.g., a business only wants 3–5 segments, not 10)?

Write 8–12 sentences.

---

## Day 5 – Interpreting Clusters & Building Simple Customer Personas

**Objectives:**
- Practice **interpreting** clusters in a structured way.
- Translate numeric summaries into meaningful “personas”.

### 1. Notebook

Create: `week8_day5_cluster_interpretation.ipynb`.

Decide on a final k (from Day 4, e.g., k=3 or 4).

Re‑fit K‑means:

```python
from sklearn.cluster import KMeans

best_k = 3  # or your chosen k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels
```

### 2. Cluster Profiles

Compute:

```python
cluster_profiles = df.groupby("cluster")[features].agg(["mean", "median", "min", "max"])
cluster_profiles
```

Also count cluster sizes:

```python
df["cluster"].value_counts(normalize=True)
```

In Markdown:

- For each cluster:
  - Describe average feature values.
  - Compare to overall means.
  - Highlight standout characteristics (e.g., “Cluster 1 is older and lower spending”).

### 3. Visualization by Cluster

Plot distributions of key features by cluster:

- Boxplots:

```python
import seaborn as sns
import matplotlib.pyplot as plt

for feat in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="cluster", y=feat)
    plt.title(f"{feat} by Cluster")
    plt.show()
```

Write 1–2 sentences per key feature describing differences across clusters.

### 4. Build Simple Personas

For each cluster, write a short textual persona:

Example:

- “Cluster 0: **Young, High‑Spending Shoppers**
  - Age: younger than average.
  - Income: medium‑high.
  - Spending score: high.
  - Business idea: target with premium offers and loyalty programs.”

Do this for each cluster.

### 5. Thinking Challenge

- If you were a marketing manager, how might you **use** these segments?
  - Which cluster might you want to acquire more of?
  - Which might be at risk of churn?
- What **limitations** should you be aware of when interpreting these unsupervised clusters (e.g., no guarantee they align with true behavior segments, sensitive attributes, etc.)?

Write 10–15 sentences.

---

## Day 6 – PCA in More Depth: Explained Variance & High‑Dimensional Data

**Objectives:**
- Understand explained variance in PCA.
- Use PCA on a higher‑dimensional dataset (possibly one you used in earlier weeks).
- See how PCA can simplify inputs for clustering or visualization.

### 1. Notebook

Create: `week8_day6_pca_deeper.ipynb`.

### 2. Apply PCA to a Higher‑Dimensional Dataset

Pick a dataset with more numeric features, for example:

- California Housing (again).
- Your Week 7 new dataset (if it has many numeric features).

Example with California Housing:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=["MedHouseVal"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Fit PCA and Check Explained Variance

```python
pca = PCA()
pca.fit(X_scaled)

explained_var = pca.explained_variance_ratio_
explained_var
```

Plot cumulative explained variance:

```python
cum_var = np.cumsum(explained_var)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()
```

In Markdown:

- How many components do you need to capture:
  - ~80% of variance?
  - ~95% of variance?

### 4. Use PCA‑Reduced Features for Clustering (Optional but Useful)

Choose, say, first 2 or 3 principal components:

```python
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
```

Run K‑means on `X_pca_2`:

```python
from sklearn.cluster import KMeans

kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels_pca = kmeans_pca.fit_predict(X_pca_2)
```

Visualize:

```python
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=labels_pca, cmap="Set1", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-means Clustering on PCA-Reduced Data")
plt.show()
```

### 5. Thinking Challenge

In Markdown:

- What are potential benefits of doing PCA **before** clustering?
- What are potential downsides (e.g., loss of interpretability, risk of discarding useful variance)?
- In what situations (dataset characteristics) would you consider applying PCA before clustering?

Write 8–12 sentences.

---

## Day 7 – Week 8 Mini‑Project: Unsupervised Segmentation & Visualization

**Objectives:**
- Pull together clustering and PCA into a small, coherent project.
- Practice explaining unsupervised results.

### 1. Notebook

Create: `week8_unsupervised_segmentation_project.ipynb`.

You can choose:
- **Option A (Recommended):** Use the **Mall Customers** dataset (clean, intuitive).
- **Option B:** Use your Week 7 new dataset or another customer‑like dataset.

### 2. Project Structure

Use Markdown headings:

1. **Introduction**
   - Describe dataset (what observations represent, what features you have).
   - Explain that you’ll perform **unsupervised segmentation** using K‑means + PCA.
   - State goals: discover a small number of customer segments and interpret them.

2. **Data Preparation**
   - Load data and show `head()` and `info()`.
   - Select numeric features for clustering.
   - Handle missing values briefly (document choices).
   - Scale features with StandardScaler.

3. **Choosing Number of Clusters**
   - Run Elbow method (plot inertia vs k).
   - Compute silhouette scores vs k.
   - Choose a reasonable k and justify in Markdown (combine numeric and domain reasoning).

4. **Clustering & PCA**
   - Run K‑means with chosen k on scaled features.
   - Compute basic cluster sizes and summary stats.
   - Run PCA (2 components).
   - Visualize clusters in PCA space (scatter plot colored by cluster).

5. **Cluster Interpretation**
   - For each cluster, compute and show:
     - Mean/median of key features.
   - Create short textual personas for each cluster.
   - Discuss:
     - Which clusters might be most valuable.
     - Any interesting/surprising patterns.

6. **Optional: Compare Raw vs PCA‑Based Clustering**
   - If you have time:
     - Compare K‑means on original features vs on PCA‑reduced features.
     - Comment on differences in cluster shapes/interpretations.

7. **Conclusions & Limitations**
   - Summarize:
     - What meaningful segments you found.
     - How a business might use them.
   - Limitations:
     - No ground truth labels.
     - Sensitivity to feature scaling and choice of features.
     - Risk of overinterpreting clusters.

### 3. Thinking / Stretch Tasks

Optional but recommended:

- **Stability Check:**
  - Run K‑means 2–3 times with different random states.
  - Do clusters look similar (e.g., similar means for each cluster)?  
  - What does this say about **stability**?
- **Try Different k:**
  - Re‑run clustering with k+1 or k‑1.
  - Are new clusters really meaningful, or just splitting existing segments without clear benefit?

### 4. Non‑Technical Summary

Write a 8–12 sentence summary aimed at a non‑technical stakeholder:

- What you did (in plain language).
- How many segments you found and what they broadly represent.
- How this could inform marketing/operations decisions.
- Caveats: why this is “exploratory,” not a final truth.

---

## After Week 8: What You Should Be Able to Do

You should now:

- Understand the mindset shift to **unsupervised learning**.
- Use **K‑means** for clustering and interpret clusters.
- Use **PCA** for dimensionality reduction and visualization.
- Apply clustering + PCA to a realistic dataset and produce a coherent, business‑oriented write‑up.

From here, natural next directions:

- Go deeper into **model deployment / end‑to‑end pipelines**, or
- Start introductory **deep learning** (Week 9: basic neural networks or CNNs), or
- Continue with more advanced unsupervised methods (e.g., t‑SNE, UMAP, density‑based clustering) once you’re comfortable with the basics.
