**Week 9 Goal:**  
Get a practical introduction to **neural networks**. By the end of the week you should be able to:

- Build and train a small feed‑forward neural network (MLP) on tabular data.  
- Optionally train a simple image classifier (e.g., MNIST).  
- Understand key ideas: layers, activations, loss, optimization, overfitting, and regularization.

Assume ~1.5–2 hours/day. We’ll use **Keras (TensorFlow)**; PyTorch would be analogous.

---

## Day 1 – Core Neural Network Concepts & Environment Setup

**Objectives**
- Understand the basic building blocks of neural networks.
- Install TensorFlow and create a minimal Keras script.

**Tasks**

1. **Concepts (markdown notes)**
   - New notebook: `week9_day1_nn_concepts.ipynb`.
   - In your own words (short bullets), define:
     - **Neuron / perceptron**: takes inputs, multiplies by weights, adds bias, passes through activation.
     - **Layer**: collection of neurons; input layer, hidden layers, output layer.
     - **Activation functions**:
       - ReLU: `max(0, x)` – used in hidden layers.
       - Sigmoid: outputs between 0 and 1 – often for binary classification output.
       - Softmax: converts a vector of scores into probabilities that sum to 1 (multi‑class).
     - **Loss function**:
       - Measures how bad predictions are.
       - Example: binary cross‑entropy for binary classification, MSE for regression.
     - **Optimizer**:
       - Algorithm adjusting weights to minimize loss (e.g., SGD, Adam).

2. **Install TensorFlow / Keras**
   - In your conda/venv (terminal, *not* in notebook):
     ```bash
     pip install "tensorflow>=2.15"
     ```
   - If you already have TF from earlier, you can skip or upgrade.

3. **Verify installation and minimal model**
   - New notebook: `week9_day1_tf_hello_world.ipynb`.
   ```python
   import tensorflow as tf
   from tensorflow import keras

   print(tf.__version__)

   # Simple sequential model (no training yet)
   model = keras.Sequential([
       keras.layers.Dense(8, activation="relu", input_shape=(4,)),  # 4 input features
       keras.layers.Dense(1, activation="sigmoid")
   ])

   model.summary()
   ```

4. **Mini reflection**
   - Explain what `input_shape=(4,)` means.
   - Based on the summary:
     - How many parameters (weights + biases) does each layer have?

**Outcome Day 1**
- TensorFlow/Keras installed.
- You understand the high‑level structure of a small neural network and can instantiate a model.

---

## Day 2 – MLP on Simple Tabular Classification (Breast Cancer)

**Objectives**
- Train a small feed‑forward network (MLP) on a standard tabular dataset.
- See training/validation curves for loss and accuracy.

**Tasks**

1. **New notebook**
   - `week9_day2_mlp_breast_cancer.ipynb`.

2. **Load dataset and split**
   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import numpy as np

   data = load_breast_cancer()
   X, y = data.data, data.target  # X: (n_samples, n_features)

   X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       random_state=42,
       stratify=y
   )
   ```

3. **Scale features (very important for NNs)**
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

4. **Build a simple Keras model**
   ```python
   import tensorflow as tf
   from tensorflow import keras

   n_features = X_train_scaled.shape[1]

   model = keras.Sequential([
       keras.layers.Input(shape=(n_features,)),
       keras.layers.Dense(16, activation="relu"),
       keras.layers.Dense(8, activation="relu"),
       keras.layers.Dense(1, activation="sigmoid")  # binary classification
   ])

   model.compile(
       optimizer="adam",
       loss="binary_crossentropy",
       metrics=["accuracy"]
   )

   model.summary()
   ```

5. **Train with validation split**
   ```python
   history = model.fit(
       X_train_scaled,
       y_train,
       epochs=30,
       batch_size=32,
       validation_split=0.2,
       verbose=0  # set to 1 if you want to see logs
   )
   ```

6. **Plot training curves**
   ```python
   import matplotlib.pyplot as plt

   history_dict = history.history
   epochs = range(1, len(history_dict["loss"]) + 1)

   plt.figure(figsize=(12,4))
   plt.subplot(1,2,1)
   plt.plot(epochs, history_dict["loss"], label="Train loss")
   plt.plot(epochs, history_dict["val_loss"], label="Val loss")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend()

   plt.subplot(1,2,2)
   plt.plot(epochs, history_dict["accuracy"], label="Train acc")
   plt.plot(epochs, history_dict["val_accuracy"], label="Val acc")
   plt.xlabel("Epoch")
   plt.ylabel("Accuracy")
   plt.legend()

   plt.tight_layout()
   plt.show()
   ```

7. **Evaluate on test set**
   ```python
   test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
   print("Test accuracy:", test_acc)
   ```

8. **Mini reflection**
   - Do you see signs of overfitting (val loss increasing while train loss decreases)?
   - Is test accuracy close to validation accuracy?

**Outcome Day 2**
- You’ve trained your first neural network on tabular data and interpreted training curves.

---

## Day 3 – Regularization: Dropout, Early Stopping, Smaller Networks

**Objectives**
- Understand and apply simple regularization techniques for NNs:
  - Dropout
  - Early stopping
  - Reducing network size

**Tasks**

1. **New notebook**
   - `week9_day3_regularization_mlp.ipynb`.

2. **Concepts (markdown)**
   - Briefly describe:
     - **Overfitting** in neural nets (very low train loss, high val loss).
     - **Dropout**: randomly “turn off” some neurons during training.
     - **Early stopping**: stop training when validation metric stops improving.

3. **Rebuild model with dropout & smaller layers**
   ```python
   import tensorflow as tf
   from tensorflow import keras

   n_features = X_train_scaled.shape[1]

   model_reg = keras.Sequential([
       keras.layers.Input(shape=(n_features,)),
       keras.layers.Dense(16, activation="relu"),
       keras.layers.Dropout(0.3),
       keras.layers.Dense(8, activation="relu"),
       keras.layers.Dropout(0.3),
       keras.layers.Dense(1, activation="sigmoid")
   ])

   model_reg.compile(
       optimizer="adam",
       loss="binary_crossentropy",
       metrics=["accuracy"]
   )
   ```

4. **Add early stopping callback**
   ```python
   early_stopping = keras.callbacks.EarlyStopping(
       monitor="val_loss",
       patience=5,
       restore_best_weights=True
   )

   history_reg = model_reg.fit(
       X_train_scaled, y_train,
       epochs=100,
       batch_size=32,
       validation_split=0.2,
       callbacks=[early_stopping],
       verbose=0
   )
   ```

5. **Compare training curves vs previous model**
   - Plot as on Day 2.
   - Note:
     - At which epoch early stopping triggered?
     - Is validation loss more stable?

6. **Evaluate on test set**
   ```python
   test_loss_reg, test_acc_reg = model_reg.evaluate(X_test_scaled, y_test, verbose=0)
   print("Regularized model - Test accuracy:", test_acc_reg)
   ```

7. **Mini reflection**
   - Did regularization improve test performance or at least reduce overfitting?
   - If accuracy went down slightly but gap between train and val decreased, is that still a win?

**Outcome Day 3**
- You can apply dropout and early stopping and see their effects on overfitting.

---

## Day 4 – Intro to Image Data & MNIST

**Objectives**
- Learn how image data is represented for NNs.
- Load and inspect the MNIST digits dataset.

**Tasks**

1. **New notebook**
   - `week9_day4_mnist_intro.ipynb`.

2. **Load MNIST from Keras**
   ```python
   from tensorflow import keras
   (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

   X_train.shape, y_train.shape
   ```

   - You should see:
     - `X_train`: (60000, 28, 28)
     - `X_test`: (10000, 28, 28)

3. **Visualize some digits**
   ```python
   import matplotlib.pyplot as plt

   plt.figure(figsize=(5,5))
   for i in range(9):
       plt.subplot(3,3,i+1)
       plt.imshow(X_train[i], cmap="gray")
       plt.title(f"Label: {y_train[i]}")
       plt.axis("off")
   plt.tight_layout()
   plt.show()
   ```

4. **Preprocess images**
   - Scale pixel values to [0,1] and flatten them for a simple MLP:
   ```python
   X_train = X_train.astype("float32") / 255.0
   X_test = X_test.astype("float32") / 255.0

   # Flatten 28x28 -> 784
   X_train_flat = X_train.reshape(-1, 28*28)
   X_test_flat = X_test.reshape(-1, 28*28)
   ```

   - Optionally, create a validation set:
   ```python
   X_val_flat = X_train_flat[:10000]
   y_val = y_train[:10000]
   X_train_flat_small = X_train_flat[10000:]
   y_train_small = y_train[10000:]
   ```

5. **Mini reflection**
   - What is the shape of a single image before and after flattening?
   - Why is scaling to [0,1] helpful for neural nets?

**Outcome Day 4**
- You can load, visualize, and preprocess image data for neural networks.

---

## Day 5 – MLP for MNIST Digit Classification

**Objectives**
- Build and train a simple feed‑forward neural network on MNIST.
- Evaluate its accuracy.

**Tasks**

1. **New notebook**
   - `week9_day5_mlp_mnist.ipynb` (or continue from Day 4).

2. **Define the model**
   ```python
   from tensorflow import keras

   model = keras.Sequential([
       keras.layers.Input(shape=(28*28,)),
       keras.layers.Dense(256, activation="relu"),
       keras.layers.Dense(128, activation="relu"),
       keras.layers.Dense(10, activation="softmax")  # 10 classes
   ])

   model.compile(
       optimizer="adam",
       loss="sparse_categorical_crossentropy",
       metrics=["accuracy"]
   )

   model.summary()
   ```

3. **Train**
   ```python
   history = model.fit(
       X_train_flat_small, y_train_small,
       epochs=10,
       batch_size=128,
       validation_data=(X_val_flat, y_val),
       verbose=1
   )
   ```

4. **Plot training curves (loss & accuracy)**
   ```python
   import matplotlib.pyplot as plt

   history_dict = history.history
   epochs = range(1, len(history_dict["loss"]) + 1)

   plt.figure(figsize=(12,4))
   plt.subplot(1,2,1)
   plt.plot(epochs, history_dict["loss"], label="Train loss")
   plt.plot(epochs, history_dict["val_loss"], label="Val loss")
   plt.legend()
   plt.xlabel("Epoch")
   plt.ylabel("Loss")

   plt.subplot(1,2,2)
   plt.plot(epochs, history_dict["accuracy"], label="Train acc")
   plt.plot(epochs, history_dict["val_accuracy"], label="Val acc")
   plt.legend()
   plt.xlabel("Epoch")
   plt.ylabel("Accuracy")

   plt.tight_layout()
   plt.show()
   ```

5. **Evaluate on test set**
   ```python
   test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
   print("Test accuracy:", test_acc)
   ```

6. **Mini reflection**
   - What test accuracy did you get (~0.95+ is common for basic MLP)?
   - Are there signs of overfitting (train acc > val acc by a lot)?

**Outcome Day 5**
- You can train an MLP classifier on MNIST and interpret performance.

---

## Day 6 – Inspecting Predictions & Misclassifications

**Objectives**
- Look at individual predictions.
- Inspect misclassified examples to build intuition.

**Tasks**

1. **New notebook**
   - `week9_day6_mnist_analysis.ipynb` (or continue from Day 5).

2. **Get predictions and probabilities**
   ```python
   import numpy as np

   y_pred_proba = model.predict(X_test_flat)  # shape (10000, 10)
   y_pred = np.argmax(y_pred_proba, axis=1)
   ```

3. **Overall metrics**
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Classification report:\n", classification_report(y_test, y_pred))

   cm = confusion_matrix(y_test, y_pred)
   cm
   ```

   - Plot confusion matrix (optional):
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt

     plt.figure(figsize=(6,5))
     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
     plt.xlabel("Predicted")
     plt.ylabel("True")
     plt.title("Confusion Matrix - MNIST")
     plt.show()
     ```

4. **Look at some misclassified examples**
   ```python
   mis_idx = np.where(y_pred != y_test)[0]
   print("Number of misclassified:", len(mis_idx))

   # Show first 9 misclassified
   plt.figure(figsize=(5,5))
   for i, idx in enumerate(mis_idx[:9]):
       plt.subplot(3,3,i+1)
       plt.imshow(X_test[idx], cmap="gray")
       plt.title(f"True:{y_test[idx]}, Pred:{y_pred[idx]}")
       plt.axis("off")
   plt.tight_layout()
   plt.show()
   ```

5. **Mini reflection**
   - Are the misclassified digits ambiguous/noisy?
   - Are there specific digits (e.g., 5 vs 8, 4 vs 9) that the model confuses more often?

**Outcome Day 6**
- You can go beyond accuracy to inspect confusion matrix and visualize errors in a NN classifier.

---

## Day 7 – Week 9 Mini Project: End‑to‑End NN (Tabular or MNIST)

**Objectives**
- Consolidate Week 9 learning in one polished notebook:
  - Either a tabular NN project or MNIST (or both briefly).
- Demonstrate understanding of architecture, training, and regularization.

**Tasks**

1. **Choose focus**
   - Option A (recommended if you like images): MNIST classification.
   - Option B: Tabular classification (Breast Cancer, Titanic with preprocessed numeric features).

2. **New notebook**
   - `week9_day7_mini_project_nn.ipynb`.

3. **Suggested structure**

   ### 1. Problem & Data
   - 3–5 sentences:
     - Dataset description.
     - Task (binary/multi‑class classification).
     - Why a neural network is appropriate/interesting.

   ### 2. Data Loading & Preprocessing
   - Show shapes, sample examples (images or tabular rows).
   - For tabular:
     - Train/test split.
     - Scaling numeric features.
   - For images:
     - Scaling pixel values.
     - Any reshaping/flattening.

   ### 3. Model Architecture
   - Define your MLP model.
   - Explain in markdown:
     - Number of layers and units.
     - Choice of activations.
     - Output layer and loss function.

   ### 4. Training Procedure
   - Train with:
     - Validation split or explicit val set.
     - Early stopping (and optionally dropout).
   - Plot loss and accuracy curves.
   - Comment on:
     - Overfitting signs (if any).
     - Early stopping behavior.

   ### 5. Evaluation & Error Analysis
   - Evaluate performance on test set.
   - Report:
     - Accuracy (and other metrics if tabular).
     - Confusion matrix.
   - Show:
     - Some misclassifications (especially for MNIST).
   - Bullet points (5–8) interpreting results:
     - Strengths and weaknesses.
     - Common error types.

   ### 6. Reflection & Next Steps
   - 4–6 bullets:
     - What changed when you added regularization?
     - How does this NN compare to a tree‑based model you trained earlier on similar data (if applicable)?
     - Ideas to improve:
       - Deeper or wider network?
       - CNN for images instead of MLP?
       - More tuning and data augmentation (for images)?

4. **Polish**
   - Clean code and comments.
   - Clear headings and text.
   - Ensure notebook runs top‑to‑bottom without errors.

**Outcome Day 7**
- A coherent neural network mini‑project showing you can:
  - Build, train, and regularize simple NNs.
  - Interpret training curves and evaluation metrics.
  - Analyze model errors.
- You’re ready in Week 10 to think about end‑to‑end workflows, interpretability, and possibly more advanced NN architectures (e.g., CNNs) if you choose.
