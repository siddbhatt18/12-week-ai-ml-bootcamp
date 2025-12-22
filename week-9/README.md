Here is your 7‑day plan for **Week 9**, focused on an introductory but practical taste of **deep learning** with **neural networks** for supervised learning.

We’ll use **Keras (TensorFlow)**, because it’s high‑level and beginner‑friendly. You’ll train a basic **feed‑forward network** for classification (MNIST or Fashion‑MNIST) and, optionally, a simple network for tabular data.

Assume ~1.5–2.5 hours/day.

---

## Overall Week 9 Goals

By the end of this week, you should be able to:

- Explain at a high level what a **neural network** is (layers, activations, loss, optimizer).
- Use **Keras** to define, train, and evaluate a simple fully connected network.
- Train a small image classifier (e.g., MNIST/Fashion‑MNIST).
- Interpret training curves (loss/accuracy vs epochs) and recognize **overfitting**.
- Apply basic regularization (dropout, early stopping).
- Compare neural nets to traditional ML models conceptually.

---

## Day 1 – Neural Network Concepts & Environment Setup

**Objectives:**
- Get a conceptual understanding of neural networks.
- Ensure TensorFlow/Keras is installed and works.

### 1. Notebook

Create: `week9_day1_nn_intro.ipynb`.

### 2. Conceptual Notes (Markdown)

In your own words (no copying), write short sections for:

1. **Perceptron / Neuron**
   - Takes inputs (numbers), multiplies by weights, adds bias.
   - Passes result through an **activation function** (e.g., ReLU, sigmoid).

2. **Layer**
   - A group of neurons; each layer’s output is input for next layer.
   - **Input layer** → **hidden layers** → **output layer**.

3. **Forward pass**
   - Data flows from input to output.
   - Network produces predictions.

4. **Loss function**
   - Measures how wrong predictions are.
   - Example:
     - Classification: cross‑entropy.
     - Regression: MSE.

5. **Training / Backpropagation (high level)**
   - Compute loss.
   - Compute gradients of loss w.r.t weights.
   - Update weights using **optimizer** (e.g., SGD, Adam) to reduce loss.

6. **Epoch / Batch**
   - Epoch: one full pass over training data.
   - Mini‑batch: subset of data to compute gradients.

You don’t need mathematical details—focus on intuition.

### 3. Install & Test TensorFlow/Keras

In a terminal or notebook cell:

```python
import tensorflow as tf
print(tf.__version__)
```

If import fails:
- Search: `install tensorflow conda` or `pip install tensorflow`.
- After installation, re‑run the import.

### 4. Quick Sanity Check: Define a Tiny Model

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(4,)),
    layers.Dense(1, activation="sigmoid")
])

model.summary()
```

In Markdown:
- Interpret `input_shape=(4,)` (4‑dimensional input).
- Interpret final layer with 1 neuron + sigmoid for binary output (0–1 probability).

### 5. Thinking Challenge

Write 8–10 sentences answering:

- How are neural networks similar to linear/logistic regression?
- How are they more flexible (non‑linear) than simple linear models?
- What might be trade‑offs (e.g., more data needed, less interpretable)?

---

## Day 2 – First Neural Network on Tabular Data (Binary Classification)

**Objectives:**
- Build and train a simple dense network on a **tabular binary classification** task.
- Get used to Keras training loop and metrics.

Use a dataset you already know (e.g., Titanic) or a simple binary Kaggle dataset. Below, I’ll assume Titanic.

### 1. Notebook

Create: `week9_day2_tabular_nn_titanic.ipynb`.

### 2. Data Preparation

Reuse your Titanic preprocessing pipeline, but aim to end up with:

- `X_train`, `X_val`, `X_test` (or at least train/test).
- `y_train`, `y_val`, `y_test`.

If you only have train/test, you can split train further:

```python
from sklearn.model_selection import train_test_split

# After you build X_encoded and y
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
```

Convert to floats and numpy arrays:

```python
import numpy as np

X_train = X_train.values.astype("float32")
X_val = X_val.values.astype("float32")
X_test = X_test.values.astype("float32")

y_train = y_train.values.astype("float32")
y_val = y_val.values.astype("float32")
y_test = y_test.values.astype("float32")
```

### 3. Define a Simple Model

```python
from tensorflow import keras
from tensorflow.keras import layers

input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # binary output
])

model.summary()
```

### 4. Compile & Train

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0  # you can set to 1 if you want to see logs
)
```

### 5. Plot Training Curves

```python
import matplotlib.pyplot as plt

history_dict = history.history

plt.figure(figsize=(6,4))
plt.plot(history_dict["loss"], label="train_loss")
plt.plot(history_dict["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history_dict["accuracy"], label="train_acc")
plt.plot(history_dict["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
```

### 6. Evaluate on Test Set

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
```

Compare to your logistic regression test accuracy.

### 7. Thinking Challenge

In Markdown:

- Does the network overfit (training accuracy much higher than validation)?
- Around which epoch does validation loss stop decreasing (early stopping point)?
- How does test accuracy compare to logistic regression? Any major difference?

Write 8–12 sentences.

---

## Day 3 – MNIST or Fashion‑MNIST: Simple Image Classification (Dense Network)

**Objectives:**
- Work with image data.
- Train a neural network to classify digits (MNIST) or clothing items (Fashion‑MNIST).

### 1. Notebook

Create: `week9_day3_mnist_dense.ipynb`.

### 2. Load Dataset

Use built‑in Keras dataset (choose one):

```python
from tensorflow import keras

# Option A: MNIST digits
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Option B: Fashion-MNIST (recommended, slightly more interesting)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

Check shapes:

```python
x_train.shape, y_train.shape
```

### 3. Preprocess Data

- Scale pixel values to [0, 1].
- Flatten 28x28 images into 784‑dim vectors for a dense network.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Create validation set from train
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)
```

### 4. Define Model

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(28*28,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes
])

model.summary()
```

### 5. Compile & Train

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=0
)
```

Plot training/validation curves like yesterday.

### 6. Evaluate & Inspect Predictions

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)
```

Inspect a few predictions:

```python
import numpy as np
import matplotlib.pyplot as plt

pred_probs = model.predict(x_test[:10])
pred_labels = pred_probs.argmax(axis=1)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_test[i]}, Pred: {pred_labels[i]}")
    plt.axis("off")
    plt.show()
```

### 7. Thinking Challenge

- Is your network overfitting, underfitting, or roughly okay?
- What might happen if you:
  - Add more layers/neurons?
  - Train for many more epochs?
- How does this performance compare to what a simple logistic regression on raw pixels might do (conceptually)?

Write 8–12 sentences.

---

## Day 4 – Overfitting, Dropout & Early Stopping

**Objectives:**
- See how to combat overfitting in neural networks.
- Apply **dropout** and **early stopping** to your Fashion‑MNIST model.

### 1. Notebook

Create: `week9_day4_regularization_nn.ipynb`.

Load and preprocess Fashion‑MNIST as on Day 3 (or reuse the notebook structure).

### 2. Baseline Model (Re‑Run, Shorter)

Train a baseline for fewer epochs (e.g., 20) and record training/validation curves.

### 3. Add Dropout

Modify model:

```python
from tensorflow import keras
from tensorflow.keras import layers

model_dropout = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(28*28,)),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
```

Compile as before.

Train:

```python
history_dropout = model_dropout.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=0
)
```

Plot train/val loss & accuracy for baseline vs dropout side‑by‑side.

### 4. Add Early Stopping

Use Keras callback:

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,  # epochs with no improvement before stopping
    restore_best_weights=True
)

history_es = model_dropout.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=0
)
```

Check:

- How many epochs did it actually train?
- Test accuracy.

### 5. Thinking Challenge

In Markdown:

- Compare:
  - Baseline (no dropout, no early stopping).
  - Dropout only.
  - Dropout + early stopping.
- Which combination gives the best **generalization** (validation/test accuracy, stable loss)?
- Conceptually:
  - Why does dropout help with overfitting (randomly disabling neurons)?
  - Why is early stopping like a form of regularization?

Write 10–15 sentences.

---

## Day 5 – Neural Networks vs Traditional ML on Tabular Data

**Objectives:**
- Compare dense neural networks to your existing models on a **tabular** dataset.
- See when NNs might or might not be worth it on tabular data.

### 1. Notebook

Create: `week9_day5_tabular_nn_vs_trees.ipynb`.

Choose one tabular dataset you know well, e.g.:

- California Housing (regression), or
- Titanic or your Week 7 dataset (classification).

Below assumes classification (Titanic or your new dataset).

### 2. Prepare Data

Use the same **train/val/test split** and preprocessing as for your best traditional model:

- `X_train, X_val, X_test`
- `y_train, y_val, y_test`

Convert to numpy arrays and scale numeric features if you haven’t.

### 3. Dense NN for Tabular

Define a slightly more careful model:

```python
from tensorflow import keras
from tensorflow.keras import layers

input_dim = X_train.shape[1]

tab_model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # for binary classification
])

tab_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history_tab = tab_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=0
)
```

Evaluate on test set (accuracy, maybe compute precision/recall/F1 with scikit‑learn).

### 4. Compare to Best Traditional Model

You already have:

- Logistic regression results.
- Random forest.
- Gradient boosting.

Create a small comparison table:

- Model type.
- Test accuracy.
- Other metrics (F1, ROC AUC, or RMSE for regression).

### 5. Thinking Challenge

In Markdown:

- Does the NN outperform your best tree/boosting model?
- Even if it’s similar, what are pros/cons of using an NN instead of tree‑based methods on tabular data?
- When might you **prefer**:
  - Trees/ensembles?
  - Neural nets?

Write 10–15 sentences.

---

## Day 6 – Understanding & Debugging Training: Learning Rates & Batch Sizes

**Objectives:**
- Develop intuition about **learning rate** and **batch size**.
- See how they affect training stability and speed.

Use the Fashion‑MNIST dense model again (or a smaller subset to speed up experiments).

### 1. Notebook

Create: `week9_day6_lr_batch_experiments.ipynb`.

Load and preprocess Fashion‑MNIST as previously (flatten to 784, scale to [0,1]).

### 2. Experiment with Learning Rates

Define a function to build a simple model:

```python
def build_model(lr=0.001):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(28*28,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

Try different learning rates:

```python
lrs = [0.0001, 0.001, 0.01]
histories = {}

for lr in lrs:
    print(f"Training with lr={lr}")
    model = build_model(lr=lr)
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=0
    )
    histories[lr] = history.history
```

Plot val_loss per epoch for each lr on same figure.

Interpret:

- Very small lr: slow, might not reach good loss in few epochs.
- Very large lr: may diverge or be unstable.

### 3. Experiment with Batch Sizes

Fix a decent learning rate (e.g., 0.001). Try batch sizes: 32, 128, 512.

Compare:

- Training time per epoch (rough sense).
- Final validation accuracy after a fixed number of epochs.

### 4. Thinking Challenge

In Markdown:

- How does learning rate affect:
  - Convergence speed.
  - Stability of training (loss oscillations, divergence)?
- How does batch size affect:
  - Epoch time.
  - Generalization (sometimes smaller batches generalize slightly better)?
- If training is unstable (val_loss jumping wildly), what might you tweak first?

Write 10–15 sentences.

---

## Day 7 – Week 9 Mini‑Project: End‑to‑End NN Classification (Images or Tabular)

**Objectives:**
- Build and document a **small NN project** end‑to‑end.
- Practice explaining architecture, training, and evaluation.

You can choose:

- **Option A (Recommended):** Fashion‑MNIST image classifier (full project).
- **Option B:** Tabular binary classification (e.g., Titanic or your Week 7 dataset).

I’ll assume Option A; adapt accordingly.

### 1. Notebook

Create: `week9_nn_classification_project.ipynb`.

### 2. Project Structure (Markdown + Code)

Sections:

1. **Introduction**
   - Problem: classify clothing images into 10 categories (Fashion‑MNIST).
   - Why interesting: simple, standard benchmark to practice NN basics.
   - Goal: achieve high test accuracy (>85–90%) with a small dense or simple CNN model.

2. **Data Loading & Preprocessing**
   - Load Fashion‑MNIST.
   - Show some example images with labels.
   - Preprocess:
     - Scale pixels to [0,1].
     - Split into train/val/test.
     - Flatten (if using dense network) or keep as 28x28x1 (if using simple CNN, optional stretch).

3. **Model Architecture**
   - Define your model (dense or small CNN).
   - Explain:
     - Number and type of layers.
     - Activations.
     - Output layer and loss function.

4. **Training Setup**
   - Compile model:
     - Optimizer.
     - Loss.
     - Metrics.
   - Use callbacks:
     - EarlyStopping (and Dropout layers if desired).
   - Train model; capture `history`.

5. **Results**
   - Plot training & validation loss/accuracy.
   - Evaluate on test set.
   - Show a few correct and incorrect predictions with images.
   - Compute confusion matrix for deeper insight (optional).

6. **Discussion**
   - Overfitting signs? Did dropout/early stopping help?
   - Where does the model still make errors (e.g., confusing similar classes)?
   - How does this compare to simpler models (conceptually) like logistic regression on pixels?

7. **Conclusions & Next Steps**
   - Summarize:
     - Final test accuracy.
     - What architectural or training changes helped the most.
   - Next ideas:
     - Try a simple CNN.
     - Data augmentation.
     - Hyperparameter tuning.

### 3. Thinking / Stretch Tasks

Pick at least one:

- **Try a simple CNN** (if you only used dense so far):

  ```python
  model_cnn = keras.Sequential([
      layers.Reshape((28, 28, 1), input_shape=(28*28,)),
      layers.Conv2D(32, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation="relu"),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(64, activation="relu"),
      layers.Dense(10, activation="softmax")
  ])
  ```

  See if accuracy improves.

- **Error Analysis:**
  - Show a grid of images the model misclassifies.
  - Look for patterns (similar shapes, low contrast, etc.).

### 4. Non‑Technical Summary

Write a 8–12 sentence summary for a non‑technical audience:

- What you built (an image classifier).
- Rough accuracy (“X out of 100 images are correctly classified”).
- What the network learned (in intuitive terms—shapes/patterns).
- Strengths and limitations (e.g., might fail on very unusual images).

---

## After Week 9: What You Should Be Able to Do

You should now:

- Understand basic **neural network concepts** and how they relate to traditional models.
- Build, train, and evaluate simple NNs in **Keras** for:
  - Tabular data.
  - Simple image classification.
- Use **dropout** and **early stopping** to fight overfitting.
- Read learning curves and debug training via learning rate/batch size tweaks.
- Write a coherent NN project notebook from data → model → evaluation → insights.

From here, natural next steps:

- Week 10: more on **pipelines, hyperparameter tuning, and model comparison** (including NNs).
- Or go deeper into **CNNs for images** or **NLP with embeddings/RNNs/Transformers**, depending on your interest.
