# DL_PR2 — Deep Learning Practical Report 2
### MLP · Activation Functions · Weight Initialization · Loss Functions · Batch Normalization · Optimizers

**Institute:** Red & White Skill Education
**Subject:** Deep Learning
**Dataset:** Adult Income (Census Income) — [UCI](https://archive.ics.uci.edu/dataset/2/adult) · [Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
**Task:** Binary classification — predict whether an individual's annual income exceeds $50K
**Framework:** TensorFlow 2.x / Keras + scikit-learn (Python 3.x, Jupyter Notebook)

---

## 📊 1. Dataset

The Adult Income (Census Income) dataset is drawn from the 1994 US Census database. It contains
48,842 records with a mix of numeric and categorical demographic/employment features. The task is
to predict whether a person earns more or less than $50,000/year.

| Field | Detail |
|---|---|
| Source | UCI Machine Learning Repository / Kaggle (Kaggle version used — cleaner, no trailing periods in `income`) |
| Rows | 48,842 (45,222 after cleaning) |
| Features | 12 input features after dropping `fnlwgt` and `education` |
| Target | `income` — `<=50K` (0) or `>50K` (1) |
| Class balance | ~75% `<=50K` / ~25% `>50K` — imbalanced |

**Preprocessing pipeline:**
1. Stripped whitespace from all string columns
2. Replaced `'?'` with `NaN` and dropped rows with any missing value (48,842 → 45,222 rows)
3. Dropped `fnlwgt` (census sampling weight, not predictive) and `education` (redundant with `educational-num`)
4. Encoded target: `>50K` → 1, `<=50K` → 0
5. One-hot encoded all categorical columns (`pd.get_dummies(drop_first=True)`) → 80 columns
6. Applied `StandardScaler` to numeric columns only — never to one-hot binary columns
7. Stratified 80/20 train/test split (`stratify=y`) to preserve class ratio in both sets (36,177 train / 9,045 test)

![Income Class Distribution](<plots/Income Class Distribution.png>)

*~75.2% of individuals earn ≤50K, ~24.8% earn >50K — this imbalance is why F1 / Precision / Recall are tracked alongside accuracy throughout this project.*

---

## 🔍 2. Exploratory Data Analysis

**Age vs Income**

![Age Distribution by Income](<plots/Age Distribution by Income.png>)

Higher earners cluster in the 35–55 age range, while the `<=50K` group skews younger.

**Hours Worked vs Income**

![Hours per Week by Income](<plots/Hours per Week by Income.png>)

`>50K` earners work more hours/week on average (higher median), with less spread than the `<=50K` group.

**Education Level vs Income**

![Education Level by Income](<plots/Education Level by Income.png>)

Higher education levels are associated with a much higher share of `>50K` earners.

**Correlation Heatmap**

![Correlation Heatmap](<plots/Correlation Heatmap.png>)

`educational-num` and `hours-per-week` show the strongest positive correlation with income among numeric features, followed by `age`. No two numeric features are strongly collinear with each other.

---

## 🧠 3. The `build_ann()` Function

Every experiment in this project is built on one reusable function, so every comparison changes
only a single setting at a time (a controlled experiment):

| Parameter | Default | Description |
|---|---|---|
| `input_dim` | — | Number of input features (required) |
| `hidden_units` | `[128, 64]` | Sizes of the two hidden Dense layers |
| `activation` | `'relu'` | Hidden layer activation function |
| `initializer` | `'glorot_uniform'` | Weight initialization scheme |
| `use_batch_norm` | `False` | Whether to insert BatchNormalization before each activation |
| `optimizer` | `'adam'` | Optimizer used at compile time |
| `loss` | `'binary_crossentropy'` | Loss function used at compile time |

Output layer is always a single sigmoid neuron (binary classification).

---

## 🏗️ 4. Baseline ANN

Architecture: `Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)`, trained 50 epochs.

![Baseline ANN Loss](<plots/Baseline ANN - Loss.png>)

![Baseline ANN Accuracy](<plots/Baseline ANN - Accuracy.png>)

![Confusion Matrix Baseline](<plots/Confusion Matrix - Baseline.png>)

| Metric | Value |
|---|---|
| Test Accuracy | 0.8417 |
| Precision (class 1, `>50K`) | 0.7371 |
| Recall (class 1) | 0.5616 |
| F1-score (class 1) | 0.6375 |
| ROC-AUC | 0.8891 |

---

## ⚡ 5. Activation Functions

Compared ReLU, Tanh, Sigmoid, and ELU on identical architectures (50 epochs each).

![Activation Comparison](<plots/accuracy vs val accuracy of 4 models with same parameters diff. act func.png>)

| Activation | Final Val Accuracy |
|---|---|
| ReLU | 0.8452 |
| Tanh | 0.8554 |
| Sigmoid | 0.8546 |
| **ELU** | **0.8535** (best test F1 — see full results table in §10) |

**Dead ReLU neuron check** — fraction of first-layer neurons stuck outputting 0 for a test batch
(**63.2%** of neurons were dead in this run):

![ReLU Dead Neuron Histogram](<plots/ReLU First-Layer Activation Distribution.png>)

---

## 🎯 6. Weight Initialization

Compared Glorot Uniform, Glorot Normal, He Uniform, He Normal, and Zeros (deliberate failure case).

![Weight Initialisation Convergence](<plots/Weight Initialisation — Convergence Speed Comparison.png>)

| Initializer | Final Val Accuracy |
|---|---|
| Glorot Uniform | 0.8394 |
| Glorot Normal | 0.8389 |
| He Uniform | 0.8452 |
| He Normal | 0.8411 |
| **Zeros** | **0.7526** ⚠️ — stuck at the majority-class baseline |

**Zeros failure demonstration** — the symmetry problem means the network never breaks out of a
single effective neuron per layer:

![Zeros Failure Demonstration](<plots/zero failure demonstration.png>)

**Weight distributions before vs after training** (He Normal vs Glorot Uniform, first hidden layer):

![Weight Distributions](<plots/weight distribution before vs after training.png>)

---

## 📉 7. Loss Functions

Compared BCE, MSE, Weighted BCE (`compute_class_weight`), and a custom Focal Loss implementation.
No standalone plot was exported for this section in the notebook — results are summarized below
and in the full comparison table in §10.

| Loss | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|
| BCE (baseline) | 0.7371 | 0.5616 | 0.6375 |
| MSE | — | — | 0.6596 |
| **Weighted BCE** | 0.5548 | **0.8082** | 0.6580 |
| Focal Loss | 0.7843 | 0.4380 | 0.5621 |

Weighted BCE trades some precision for a large recall gain (0.56 → 0.81) — valuable when missing a
genuine high earner is costlier than a false positive. Focal Loss, by contrast, pushed precision up
but recall down — it focused on "hard" examples but didn't rebalance the classes the way weighted
BCE did.

---

## 🧩 8. Batch Normalization

Canonical order: `Dense → BatchNorm → Activation`.

![Batch Normalization vs Baseline](<plots/Batch Normalization vs Baseline — Training Dynamics.png>)

| Model | Accuracy | F1 (class 1) |
|---|---|---|
| Baseline (no BN) | 0.8469 | 0.6629 |
| With BatchNorm | 0.8407 | 0.6452 |

**BatchNorm position experiment** (before vs after activation, 30 epochs):

![BatchNorm Position](<plots/BatchNorm Position — Validation Accuracy.png>)

| Position | Final Val Accuracy |
|---|---|
| Before activation (canonical) | 0.8419 |
| After activation | 0.8447 |

**Learned gamma / beta** (first BN layer, 128 neurons):

![Gamma and Beta Neurons](<plots/gamma and beta neurons.png>)

Neurons with `gamma` close to zero were effectively gated off by the network — it learned their
pre-activation values carry little predictive value for income.

---

## 🚀 9. Optimizers

Compared SGD, SGD+Momentum, RMSprop, Adam, and explicit Adam — using the strongest configuration
found so far (ReLU + He Normal + BatchNorm), 50 epochs each. The convergence plot for this run
(§7.2 in the notebook) is rendered inline in the notebook but was not exported to `plots/`; the
final validation accuracies are below.

| Optimizer | Final Val Accuracy |
|---|---|
| SGD | 0.8485 |
| **SGD + Momentum** | **0.8510** |
| RMSprop | 0.8502 |
| Adam | 0.8416 |
| Adam (explicit) | 0.8433 |

**Learning rate sensitivity — SGD vs Adam:**

![Learning Rate Sensitivity](<plots/learning rate sensitivity SGD vs Adam.png>)

Adam stays far more stable across a wide range of learning rates; SGD is much more sensitive —
too high a rate destabilizes it, too low and it barely moves.

---

## 🏆 10. Final Combined Model & Full Comparison

Best configuration: ReLU + He Normal + BatchNorm + Adam + BCE, trained 80 epochs.

| Metric | Value |
|---|---|
| Accuracy | 0.8383 |
| Precision | 0.7034 |
| Recall | 0.6008 |
| F1 | 0.6481 |
| ROC-AUC | 0.8950 |

**Full results comparison — every model trained in this project** (from the notebook's §7.6
results table; highest F1 and ROC-AUC highlighted in the notebook output):

| Model | Test Acc | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | 0.8417 | 0.7371 | 0.5616 | 0.6375 | 0.8891 |
| Best Activation (ELU) | 0.8469 | 0.7295 | 0.6075 | 0.6629 | 0.9059 |
| Best Initialiser (He Normal) | 0.8381 | 0.6926 | 0.6240 | **0.6565** | 0.8864 |
| Weighted BCE | 0.7917 | 0.5548 | **0.8082** | 0.6580 | 0.8830 |
| Focal Loss | 0.8308 | 0.7843 | 0.4380 | 0.5621 | 0.8867 |
| BatchNorm | 0.8407 | 0.7202 | 0.5843 | 0.6452 | 0.8931 |
| Final Combined | 0.8383 | 0.7034 | 0.6008 | 0.6481 | **0.8950** |

*Best Activation (ELU) achieves the highest F1(1) overall (0.6629), and Final Combined achieves
the highest ROC-AUC (0.8950).*

**ROC Curve — Baseline vs Weighted BCE vs Final Combined:** this comparison is generated live in
the notebook (§7.7, `roc_curve` from scikit-learn) but was not exported as a static image. Rerun
the corresponding notebook cell to view it; the AUC values for each of the three models are listed
in the results table above (0.8891 / 0.8830 / 0.8950 respectively).

---

## 💡 11. Key Takeaway

The single largest lever for minority-class (`>50K`) F1-score and Recall was **Weighted BCE** —
reweighting the loss to account for the ~75/25 class imbalance pushed recall up substantially
(0.56 → 0.81) because the model stopped being able to "coast" on the majority class. ELU as the
activation function gave the best overall F1/ROC-AUC among the architecture-level changes.
Architecture choices (activation, initializer, BatchNorm) mattered mainly for training stability
and convergence speed, but the imbalance-aware loss function had the biggest direct effect on
catching true high earners — the metric that matters most for this business problem.

---

## 📁 12. Repository Structure

```
.
├── adult_census_income.ipynb   # Main notebook — all 7 tasks
├── adult.csv                   # Dataset
├── plots/                      # All 17 generated figures
├── requirements.txt
└── README.md
```

---

## ⚙️ 13. Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow>=2.12.0
scikit-learn>=1.4.0
pandas
numpy
matplotlib
seaborn
```

---

## 🎥 14. Video Walkthrough

_Add your video link here (Google Drive or YouTube, unlisted) once recorded._

`[Video link — 5–10 min, face + screen, DL_PR2_YourName_GRID.mp4]`

---

*Red & White Skill Education · Since 2008 · "Shaping Skills for Scaling Higher"*
