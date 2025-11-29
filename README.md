# Solar Flare Prediction: Comparative Machine Learning Analysis

## 🌠 Project Overview

This project implements and compares the performance of four core Machine Learning (ML) algorithms—**Gradient Boosting (XGBoost)**, **Random Forest (RF)**, **Support Vector Machines (SVM)**, and a **Deep Artificial Neural Network (ANN)**—for forecasting **Solar Flares**. The objective is to classify whether a major solar flare (C, M, or X class) will occur within a 24-hour window, based on observable features of active solar regions (sunspots).

A central challenge in this domain is the **severe class imbalance** inherent in space weather data (flares are rare events). This work addresses this by utilizing the **SMOTETomek hybrid resampling technique** to ensure robust and non-biased model training.

### 🎯 Key Scientific Objectives

* **Binary Classification:** Convert the multi-class flare count problem into a binary classification: "Flare Will Occur (1)" vs. "Flare Will Not Occur (0)".
* **Ensemble vs. Kernel vs. Deep Learning:** Empirically compare the predictive power of ensemble methods (RF, XGBoost) against kernel-based (SVM) and deep learning models (ANN/MLP).
* **Mitigating Bias:** Demonstrate the impact of **SMOTETomek** in improving the classification of the minority class (flares), a metric often prioritized over general accuracy in critical forecasting tasks.

### ⚙️ Methodology and Data

#### 1. Data Source and Characteristics
* **Dataset:** Solar Flare Data Set from the UCI Machine Learning Repository.
* **Total Observations:** 1,065 rows.
* **Original Imbalance:** 81.1% "No Flare" (0) vs. 18.9% "Flare" (1).
* **Features:** 13 features describing active regions, such as Modified Zurich Class (`Zurich_Class`), Spot Size (`Spot_Size`), Spot Distribution (`Spot_Distribution`), and Activity level.

#### 2. Data Preprocessing Pipeline

1. **Feature Conversion:** Categorical features (Zurich Class, Spot Size, Distribution) were converted using **One-Hot Encoding** to create a numerical matrix suitable for SVM and ANN.
2. **Splitting:** Data was split into 70% Training and 30% Testing sets, ensuring the original class distribution was maintained (**stratified sampling**).
3. **Standard Scaling:** Numerical features were normalized using `StandardScaler` (Mean = 0, Standard Deviation = 1) to prevent features with larger scales from dominating distance-based algorithms (SVM, ANN). The formula used is $z=\frac{x-\mu}{\sigma}$.
4. **Hybrid Resampling (SMOTETomek):**
    * **SMOTE:** Oversamples the minority class (Flare).
    * **Tomek Links:** Cleans noisy data points near the decision boundary.
    * **Result:** The training data becomes perfectly balanced (50% Flare, 50% No Flare).

### 📊 Results and Comparative Performance

Models were trained on the balanced training set and evaluated on the untouched test set (30% of original data).

#### Final Accuracy Comparison

| Algorithm | Accuracy (%) | Rank |
| :--- | :--- | :--- |
| **Gradient Boosting (XGBoost)** | **76.88** | 1 (Best) |
| **Random Forest** | 74.69 | 2 |
| **SVM (Optimized)** | 72.81 | 3 |
| **Artificial Neural Networks (ANN)** | 71.25 | 4 |

#### Detailed Model Insights

* **Gradient Boosting (Best Performer):** Achieved the highest accuracy by sequentially training weak learners (shallow trees) to minimize residual errors.
* **Random Forest:** Strong performance due to stability and reduced overfitting via ensemble majority voting.
* **SVM and ANN (Lower Performers):**
    * **SVM:** High cardinality of categorical features challenged hyperplane separation.
    * **ANN:** 3-layer deep architecture (100-50-25 neurons) constrained by small dataset size (~1,000 rows).

### 📈 Confusion Matrix Visualization

Individual confusion matrices and classification reports are generated for each model, emphasizing **Recall**, crucial in space weather as a missed flare (False Negative) can be catastrophic.

[Example Confusion Matrix Image Placeholder]

---

## 🛠️ Setup and Execution

This project is implemented in Python and executed within a single Jupyter Notebook.

### Prerequisites

* Python 3.10+
* Libraries: `pandas`, `numpy`, `sklearn`, `imblearn` (for SMOTETomek), `matplotlib`, `seaborn`.

### Installation and Run Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/solar-flare-prediction-ml-comparison.git
    cd solar-flare-prediction-ml-comparison
    ```
2. **Open the Notebook:** Upload `SolarFlareForecast.ipynb` to your environment (Google Colab, Jupyter Lab, etc.).
3. **Data Loading:** The notebook loads data automatically from the UCI repository.
4. **Execute:** Run all cells sequentially. The notebook will:
    * Load and preprocess the raw data (Encoding, Scaling).
    * Apply **SMOTETomek** to balance the training set.
    * Train and evaluate the four ML models.
    * Generate the comparative table, confusion matrices, and classification reports.

### Future Work Recommendations

* **Image Integration:** Use solar image data (NASA SDO) with **CNNs** for hybrid prediction.
* **Time Series Analysis:** Employ **LSTM** or other RNNs to analyze sequential data (24–48 hours) to model flares as a process.
* **Advanced Optimization:** Apply **Bayesian Optimization** or **Genetic Algorithms** to fine-tune Gradient Boosting hyperparameters.

---

## 📞 Contact

**Onur YERLİKAYA** **->** **yrlkyaonur@gmail.com**
