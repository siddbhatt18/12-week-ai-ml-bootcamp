**Week 6 Goal:**  
Get comfortable with **decision trees** and **random forests**, understand **overfitting vs generalization**, and practice diagnosing and controlling model complexity.

Use the same main tabular dataset as Week 5 (classification is slightly easier for trees, e.g., Titanic / Breast Cancer / Telco Churn). Aim for ~1.5–2 hours/day.

---

## Day 1 – Intuition for Decision Trees + First Tree Model

**Objectives**
- Understand how decision trees split data.
- Train a basic decision tree classifier on your dataset.
- See initial signs of overfitting.

**Tasks**

1. **New notebook**
   - `week6_day1_decision_tree_intro.ipynb`.

2. **Concepts (markdown)**
   - In your own words, answer:
     - How does a decision tree make decisions?
       - Repeatedly splits data based on features to create “purer” groups.
     - What is “purity” (high-level)?
       - Nodes where most samples belong to the same class.
     - Why are trees prone to overfitting?
       - They can keep splitting until they memorize the training data.

3. **Load data (with preprocessing pipeline from Week 5)**
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split

   df = pd.read_csv("data/your_dataset.csv")

   target_col = "Survived"  # or your target
   id_cols = ["PassengerId"]  # adjust
   X = df.drop(columns=[target_col] + id_cols)
   y = df[target_col]
   ```

   - Reuse `num_cols`, `cat_cols`, and `preprocessor` from Week 5 (or redefine them briefly).

4. **Build a DecisionTreeClassifier pipeline**
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.pipeline import Pipeline

   tree_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),  # ColumnTransformer
       ("model", DecisionTreeClassifier(random_state=42))
   ])
   ```

5. **Train/test split + fit**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   tree_clf.fit(X_train, y_train)

   from sklearn.metrics import accuracy_score

   y_pred_train = tree_clf.predict(X_train)
   y_pred_test = tree_clf.predict(X_test)

   print("Train accuracy:", accuracy_score(y_train, y_pred_train))
   print("Test  accuracy:", accuracy_score(y_test, y_pred_test))
   ```

6. **Mini reflection**
   - Is train accuracy much higher than test accuracy?
     - If yes, that’s a sign of overfitting.
   - Why is a “perfect” train accuracy suspicious?

**Outcome Day 1**
- You can train a basic decision tree and see how it might overfit.

---

## Day 2 – Controlling Tree Complexity (max_depth, min_samples_*)  

**Objectives**
- Learn key tree hyperparameters that control overfitting.
- Experiment with them to see effect on train vs test performance.

**Tasks**

1. **New notebook**
   - `week6_day2_tree_complexity.ipynb`.

2. **Define a helper function to train & evaluate a tree**
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.pipeline import Pipeline
   from sklearn.metrics import accuracy_score

   def eval_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1):
       tree_clf = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("model", DecisionTreeClassifier(
               random_state=42,
               max_depth=max_depth,
               min_samples_split=min_samples_split,
               min_samples_leaf=min_samples_leaf
           ))
       ])
       tree_clf.fit(X_train, y_train)
       y_pred_train = tree_clf.predict(X_train)
       y_pred_test = tree_clf.predict(X_test)
       train_acc = accuracy_score(y_train, y_pred_train)
       test_acc = accuracy_score(y_test, y_pred_test)
       return train_acc, test_acc
   ```

3. **Vary max_depth**
   ```python
   depths = [None, 2, 3, 5, 10]
   results = []

   for d in depths:
       train_acc, test_acc = eval_tree(max_depth=d)
       results.append((d, train_acc, test_acc))

   import pandas as pd
   pd.DataFrame(results, columns=["max_depth", "train_acc", "test_acc"])
   ```

4. **Plot train vs test accuracy vs depth (optional but useful)**
   ```python
   import matplotlib.pyplot as plt

   df_res = pd.DataFrame(results, columns=["max_depth", "train_acc", "test_acc"])
   plt.plot(df_res["max_depth"], df_res["train_acc"], label="Train", marker="o")
   plt.plot(df_res["max_depth"], df_res["test_acc"], label="Test", marker="o")
   plt.xlabel("max_depth (None = full)")
   plt.ylabel("Accuracy")
   plt.legend()
   plt.title("Effect of max_depth on accuracy")
   plt.show()
   ```

5. **Experiment with min_samples_leaf**
   ```python
   leaves = [1, 2, 5, 10, 20]
   results_leaf = []

   for leaf in leaves:
       train_acc, test_acc = eval_tree(max_depth=None, min_samples_leaf=leaf)
       results_leaf.append((leaf, train_acc, test_acc))

   pd.DataFrame(results_leaf, columns=["min_samples_leaf", "train_acc", "test_acc"])
   ```

6. **Mini reflection**
   - Which settings appear overfitted (high train, lower test)?
   - Which configurations seem underfitted (both train and test low)?
   - Which seems like a good balance?

**Outcome Day 2**
- You see how tree hyperparameters shape bias–variance trade‑off.

---

## Day 3 – Visualizing and Interpreting a Decision Tree

**Objectives**
- Visualize a (small) decision tree.
- Interpret splits and understand how rules correspond to feature conditions.

**Tasks**

1. **New notebook**
   - `week6_day3_tree_visualization.ipynb`.

2. **Train a small tree (to keep it readable)**
   ```python
   from sklearn.tree import DecisionTreeClassifier

   small_tree_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", DecisionTreeClassifier(
           random_state=42,
           max_depth=3,       # small depth for visualization
           min_samples_leaf=5
       ))
   ])

   small_tree_clf.fit(X_train, y_train)
   ```

3. **Extract the fitted DecisionTree from the pipeline**
   ```python
   tree_model = small_tree_clf.named_steps["model"]
   ```

4. **Get feature names after preprocessing**
   ```python
   # numeric + one-hot categorical (as in Week 5)
   ohe = small_tree_clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
   cat_feature_names = ohe.get_feature_names_out(cat_cols)
   import numpy as np
   feature_names = np.concatenate([num_cols, cat_feature_names])
   ```

5. **Visualize tree structure**
   - Option A: `sklearn.tree.plot_tree`
     ```python
     from sklearn import tree
     import matplotlib.pyplot as plt

     plt.figure(figsize=(20, 10))
     tree.plot_tree(
         tree_model,
         feature_names=feature_names,
         class_names=["0", "1"],  # adjust labels
         filled=True,
         rounded=True,
         max_depth=3
     )
     plt.show()
     ```

6. **Interpret nodes**
   - Pick one non‑root node and note:
     - The feature used (e.g., `Age` or a one‑hot category).
     - The threshold.
     - What it means in plain language.
       - Example: “If Age <= 12.5 and Sex_female = 1 → likely survived”.

7. **Mini reflection**
   - Do splits intuitively make sense given your knowledge of the data?
   - How explainable are decision trees relative to linear/logistic models?

**Outcome Day 3**
- You can visualize a decision tree and read its rules in human terms.

---

## Day 4 – Random Forests: Basics & Comparison To Single Tree

**Objectives**
- Train a **RandomForestClassifier** with your existing preprocessing.
- Compare performance and overfitting vs a single tree.
- Understand bagging and ensemble idea conceptually.

**Tasks**

1. **New notebook**
   - `week6_day4_random_forest_intro.ipynb`.

2. **Concepts (markdown)**
   - In your own words:
     - A random forest is:
       - An ensemble of decision trees.
       - Each tree trained on a bootstrap sample of the data.
       - Each split considers a random subset of features.
     - Main effect:
       - Reduces variance (more stable/generalizable than a single tree).

3. **Build random forest pipeline**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import Pipeline

   rf_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(
           n_estimators=100,
           random_state=42,
           n_jobs=-1
       ))
   ])

   rf_clf.fit(X_train, y_train)

   from sklearn.metrics import accuracy_score
   y_pred_train = rf_clf.predict(X_train)
   y_pred_test = rf_clf.predict(X_test)

   print("RF Train accuracy:", accuracy_score(y_train, y_pred_train))
   print("RF Test  accuracy:", accuracy_score(y_test, y_pred_test))
   ```

4. **Compare to best decision tree from Day 2**
   - Using similar evaluation:
     ```python
     # assume best_tree_clf is your tuned tree pipeline from Day 2
     y_pred_tree_train = best_tree_clf.predict(X_train)
     y_pred_tree_test = best_tree_clf.predict(X_test)

     print("Tree Train acc:", accuracy_score(y_train, y_pred_tree_train))
     print("Tree Test  acc:", accuracy_score(y_test, y_pred_tree_test))
     ```

5. **Feature importances from random forest**
   ```python
   import numpy as np
   import pandas as pd

   rf_model = rf_clf.named_steps["model"]
   ohe = rf_clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
   cat_feature_names = ohe.get_feature_names_out(cat_cols)
   feature_names = np.concatenate([num_cols, cat_feature_names])

   importances = rf_model.feature_importances_
   fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
   fi.head(10)
   ```

6. **Mini reflection**
   - Is the random forest less overfit than the full-depth single tree?
   - Does it improve test accuracy?
   - Which features are most important and do they match expectations?

**Outcome Day 4**
- You can train and evaluate a random forest, and see how it typically outperforms/tames a single tree.

---

## Day 5 – Tuning Random Forest Hyperparameters

**Objectives**
- Explore key random forest hyperparameters and their effect on performance.
- Get hands-on with a small grid/random search.

**Tasks**

1. **New notebook**
   - `week6_day5_rf_hyperparameters.ipynb`.

2. **Key hyperparameters to explore**
   - `n_estimators`
   - `max_depth`
   - `min_samples_leaf`
   - `max_features` (number of features considered at each split)

3. **Manual loop over hyperparameters**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score
   import pandas as pd

   results = []
   for n_estimators in [50, 100, 200]:
       for max_depth in [None, 5, 10]:
           for min_samples_leaf in [1, 5]:
               rf_clf = Pipeline(steps=[
                   ("preprocessor", preprocessor),
                   ("model", RandomForestClassifier(
                       n_estimators=n_estimators,
                       max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf,
                       random_state=42,
                       n_jobs=-1
                   ))
               ])
               rf_clf.fit(X_train, y_train)
               y_pred_train = rf_clf.predict(X_train)
               y_pred_test = rf_clf.predict(X_test)
               train_acc = accuracy_score(y_train, y_pred_train)
               test_acc = accuracy_score(y_test, y_pred_test)
               results.append((n_estimators, max_depth, min_samples_leaf, train_acc, test_acc))

   res_df = pd.DataFrame(results, columns=["n_estimators", "max_depth", "min_samples_leaf", "train_acc", "test_acc"])
   res_df.sort_values(by="test_acc", ascending=False).head(10)
   ```

4. **Look for patterns**
   - Does increasing `n_estimators` always help?
   - How does `max_depth` affect train vs test accuracy?
   - Does larger `min_samples_leaf` reduce overfitting?

5. **Optional: use RandomizedSearchCV**
   ```python
   from sklearn.model_selection import RandomizedSearchCV
   from scipy.stats import randint

   rf_base = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
   ])

   param_distributions = {
       "model__n_estimators": randint(50, 301),
       "model__max_depth": [None, 5, 10, 20],
       "model__min_samples_leaf": randint(1, 11),
   }

   search = RandomizedSearchCV(
       rf_base,
       param_distributions=param_distributions,
       n_iter=20,
       cv=3,
       scoring="accuracy",
       random_state=42,
       n_jobs=-1,
       verbose=1
   )

   search.fit(X_train, y_train)
   print("Best params:", search.best_params_)
   print("Best CV accuracy:", search.best_score_)
   ```

6. **Mini reflection**
   - Which hyperparameter had strongest impact on test accuracy?
   - Do you see evidence that too deep trees inside the forest can still overfit?

**Outcome Day 5**
- You can systematically tune random forest hyperparameters and interpret their effects.

---

## Day 6 – Bias–Variance Trade‑Off & Cross‑Validation Diagnostics

**Objectives**
- Connect overfitting/underfitting to bias–variance trade-off.
- Use cross‑validation curves to understand model stability.

**Tasks**

1. **New notebook**
   - `week6_day6_bias_variance_cv.ipynb`.

2. **Concepts (markdown)**
   - Write concise definitions:
     - High bias model → underfitting: too simple, misses patterns.
     - High variance model → overfitting: fits noise, unstable across samples.
     - Goal: find a sweet spot between bias and variance.

3. **Use cross_val_score to compare a small vs large tree**
   ```python
   from sklearn.model_selection import cross_val_score

   small_tree_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", DecisionTreeClassifier(
           random_state=42,
           max_depth=3
       ))
   ])

   big_tree_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", DecisionTreeClassifier(
           random_state=42,
           max_depth=None
       ))
   ])

   for name, model in [("Small Tree", small_tree_clf), ("Big Tree", big_tree_clf)]:
       scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
       print(name)
       print("  Mean accuracy:", scores.mean())
       print("  Std:", scores.std())
   ```

4. **Random forest stability**
   ```python
   from sklearn.ensemble import RandomForestClassifier

   rf_clf = Pipeline(steps=[
       ("preprocessor", preprocessor),
       ("model", RandomForestClassifier(
           n_estimators=200,
           random_state=42,
           n_jobs=-1
       ))
   ])

   scores_rf = cross_val_score(rf_clf, X, y, cv=5, scoring="accuracy")
   print("Random Forest CV mean:", scores_rf.mean())
   print("Random Forest CV std:", scores_rf.std())
   ```

5. **Learning curve (optional but valuable)**
   ```python
   from sklearn.model_selection import learning_curve
   import numpy as np
   import matplotlib.pyplot as plt

   train_sizes, train_scores, val_scores = learning_curve(
       rf_clf, X, y, cv=5, scoring="accuracy",
       train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
   )

   train_mean = train_scores.mean(axis=1)
   val_mean = val_scores.mean(axis=1)

   plt.plot(train_sizes, train_mean, label="Train")
   plt.plot(train_sizes, val_mean, label="Validation")
   plt.xlabel("Training set size")
   plt.ylabel("Accuracy")
   plt.legend()
   plt.title("Learning Curve - Random Forest")
   plt.show()
   ```

6. **Mini reflection**
   - Compare small vs big tree:
     - Which has higher mean CV accuracy?
     - Which has higher std (more variance)?
   - From the learning curve:
     - Do more data points still improve validation performance?
     - Does the gap between train and validation shrink?

**Outcome Day 6**
- You can diagnose bias/variance behavior via train/test and cross‑validation, and see how random forests improve stability.

---

## Day 7 – Week 6 Mini Project: Trees & Forests End‑to‑End

**Objectives**
- Consolidate Week 6 concepts into a single, well-organized notebook:
  - Decision tree vs random forest.
  - Overfitting control.
  - Hyperparameter tuning and feature importance.

**Tasks**

1. **New notebook**
   - `week6_day7_mini_project_trees_forests.ipynb`.

2. **Structure**

   ### 1. Problem Overview
   - 3–5 sentences:
     - Dataset, target, and task (classification or regression).
     - Why tree‑based models might be a good choice.

   ### 2. Data & Preprocessing
   - Load data, define `X`, `y`.
   - Show class distribution (if classification).
   - Define `id_cols`, `num_cols`, `cat_cols`.
   - Build `preprocessor` (ColumnTransformer with imputation + OneHot, etc).

   ### 3. Baseline Models
   - Fit:
     - LogisticRegression (or LinearRegression) as a simple baseline.
     - Unrestricted DecisionTree (max_depth=None).
   - Report:
     - Train and test accuracy (or RMSE/R² for regression).
     - Clear sign if the tree is overfitting.

   ### 4. Controlled Trees
   - Train a few trees with different `max_depth` / `min_samples_leaf`.
   - Show a small table of:
     - `max_depth`, `train_acc`, `test_acc`.
   - Pick one “good trade‑off” tree and briefly explain.

   ### 5. Random Forest & Tuning
   - Train a default RandomForest.
   - Tune a small set of hyperparameters (either with manual loop or RandomizedSearchCV).
   - Compare:
     - Best tuned RF vs best tree vs baseline model.
   - Report metrics in a summary DataFrame.

   ### 6. Interpretation
   - Show top 10 random forest feature importances.
   - If feasible, show a small tree visualization (max_depth=3) to illustrate rules.
   - In markdown, 8–10 bullet points:
     - What patterns do trees capture that linear models might miss?
     - Any evidence of overfitting/underfitting in your experiments?
     - Which features are consistently important?
     - How much tuning improved over default settings?

   ### 7. Conclusions & Next Steps
   - 4–6 bullets:
     - When you might prefer trees/forests.
     - Potential next algorithms (gradient boosting, XGBoost).
     - Ideas for better features or dealing with class imbalance.

3. **Polish**
   - Ensure the notebook runs top‑to‑bottom without errors.
   - Add markdown explanations before/after main code blocks.
   - Keep plots readable with titles and axis labels.

**Outcome Day 7**
- A complete trees/forests experiment notebook comparing single trees to random forests and understanding overfitting control.
- You’re ready for Week 7, where you’ll explore **gradient boosting** and more systematic **hyperparameter tuning**.
