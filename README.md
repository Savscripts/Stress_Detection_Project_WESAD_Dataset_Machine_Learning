# Data-Driven Stress Detection for Enhancing Workplace Productivity

## 1. Introduction

Stress is widely recognized as a significant factor affecting physical health, cognitive performance, and workplace productivity. Traditional methods of stress assessment, primarily surveys or interviews, are subjective and fail to capture fluctuations throughout the day. Wearable sensors provide continuous physiological data that reflect autonomic responses, enabling an objective approach to stress detection.

This project develops an end-to-end machine learning pipeline using wrist-based signals from the WESAD dataset to classify stress versus non-stress states. The objective is to determine whether real-world deployable sensors, when paired with interpretable machine-learning models, can deliver reliable stress identification suitable for workplace wellness applications.

---

## 2. Literature Review

The WESAD dataset introduced by Schmidt et al. (2018) remains a foundational resource for stress detection research. Their multimodal study demonstrated that physiological signals such as electrodermal activity (EDA), heart-rate variability (HRV), temperature, and motion carry distinct patterns during stress, enabling machine-learning models to achieve strong performance. Their results also showed that classical machine-learning methods often match or outperform deep learning when engineered physiological features are used.

Richer et al. (2024) examined motor-behavior responses to psychosocial stress, finding that reduced movement (“freezing behavior”) accompanies physiological arousal. While their focus differed from wrist sensors, the finding reinforces the multi-system nature of stress responses.

Broader reviews (Razavi et al., 2024) highlight preprocessing quality—filtering, normalization, and artifact removal—as a critical determinant of model success. These reviews also emphasize subject variability and the need for personalized or adaptive models. Sabry et al. (2022) discuss deployment challenges in wearable healthcare systems, identifying privacy, battery efficiency, and sensor reliability as important real-world considerations.

Taken together, these studies guided the design of this project: rigorous preprocessing, physiologically meaningful feature construction, and evaluation that considers both performance and real-world feasibility.

---

## 3. Method and Implementation

### 3.1 Dataset and Labels

The study uses wrist-based Empatica E4 signals from 15 subjects in the WESAD dataset. Signals include EDA, BVP, temperature, and accelerometry. The original three labels (baseline, stress, amusement) were mapped into a binary classification task:

- **Stress = 1**
- **Non-stress = 0**

This reflects practical deployment settings, where identifying stress onset is more meaningful than distinguishing amusement from calm.

---

### 3.2 Data Processing Pipeline

All implementation steps were coded in Python using NumPy, pandas, SciPy, scikit-learn, imbalanced-learn, and TensorFlow. The pipeline was designed to accommodate varying sampling rates, missing channels, and noise typical of wearable sensor data.

**Signal Standardization**

Empatica channels operate at different native sampling frequencies (e.g., EDA/TEMP at 4 Hz, BVP at 64 Hz, ACC at 32 Hz). To create a uniform representation, all channels were resampled to 32 Hz. A Butterworth low-pass filter was applied to reduce high-frequency noise and motion artifacts.

**Label Alignment**

WESAD labels exist at a coarse temporal resolution. Labels were upsampled to match the resampled signal length using nearest-neighbor mapping. This ensured that every time point had a corresponding stress or non-stress label.

---

### 3.3 Windowing and Feature Construction

Signals were segmented into 60-second windows with 50% overlap to produce smoother transitions and increase training samples. Within each window, the majority label determined its class. This produced **3,542 usable windows** across all subjects.

For each window, **13 physiologically grounded features** were extracted:

- **EDA**
  - Mean
  - Standard deviation
  - Linear trend (slope)
  - Number of skin-conductance peaks
  - Optional phasic component

- **BVP / HRV**
  - Mean heart rate
  - RMSSD
  - SDNN

- **Temperature**
  - Mean
  - Variability
  - Change over the window

- **Motion**
  - Accelerometer magnitude standard deviation
  - Energy

This feature set draws on established biomarkers of sympathetic arousal and thermoregulatory stress responses.

---

### 3.4 Model Training Framework

Classical machine-learning models—**Random Forest (RF)**, **Support Vector Machine (SVM)**, and **Logistic Regression**—were trained using a unified preprocessing pipeline consisting of median imputation, SMOTE oversampling, feature scaling, and model fitting. The dataset was split using an **80/20 stratified train–test split** to preserve class proportions.

A **1D Convolutional Neural Network (CNN)** was also implemented to evaluate whether raw windowed sequences (EDA and BVP) provide performance advantages over engineered features. The CNN architecture included convolutional layers, batch normalization, pooling layers, dropout, and a final sigmoid classifier.

---

## 4. Experiment Setup

Performance was evaluated using **Accuracy**, **F1 Score**, and **ROC-AUC**, which together capture discrimination ability and sensitivity to class imbalance. Because stress windows were less frequent, a probability threshold search was conducted to maximize the F1 score.

Confusion matrices were used to inspect misclassification patterns, and feature importance values were extracted from the Random Forest model to assess physiological relevance.

---

## 5. Results

### Table 1. Model Performance Summary

| Model                    | Accuracy | F1 Score | ROC-AUC | Notes |
|-------------------------|----------|----------|---------|-------|
| Logistic Regression     | 0.685    | 0.444    | 0.707   | Linear baseline; weakest performance |
| SVM (RBF)               | 0.804    | 0.627    | 0.871   | Handles non-linear patterns moderately well |
| Random Forest (default) | 0.877    | 0.717    | 0.922   | Best classical model |
| Random Forest (tuned)   | 0.901    | 0.750    | —       | Improved stress recall with threshold adjustment |
| CNN                     | 0.738    | 0.205    | —       | Underperformed due to imbalance & small dataset |

The Random Forest model provided the strongest overall performance, reflecting its ability to capture nonlinear interactions among physiological features such as EDA variability, temperature changes, and motion metrics. Logistic Regression performed poorly due to its linear decision boundary, while SVM achieved moderate performance.

The CNN did not outperform classical approaches, consistent with prior findings that small, subject-mixed physiological datasets are challenging for deep learning models.

---

### Table 2. Random Forest Confusion Matrix

|                     | Predicted Non-Stress | Predicted Stress |
|---------------------|---------------------|------------------|
| **Actual Non-Stress** | 512                 | 52               |
| **Actual Stress**     | 35                  | 110              |

---

### Table 3. Top Feature Importances (Random Forest)

| Rank | Feature     | Type        | Description |
|-----:|------------|-------------|-------------|
| 1    | TEMP_delta | Temperature | Change in temperature over window |
| 2    | ACC_std    | Motion      | Variability in movement |
| 3    | EDA_std    | EDA         | Variation in skin conductance |
| 4    | EDA_mean   | EDA         | Baseline skin conductance |
| 5    | ACC_energy | Motion      | Movement intensity |

Feature importance analysis revealed that temperature and EDA-based features were the strongest predictors of stress, followed by motion-based metrics. These findings align with known physiological stress responses.

---

## 7. Conclusion

This project demonstrates that stress can be reliably detected from wrist-based physiological data using interpretable machine-learning techniques. The developed pipeline—incorporating resampling, noise filtering, feature engineering, and robust evaluation—provides a strong foundation for real-world stress monitoring.

Future work should explore subject-wise validation, personalized models, additional sensor modalities, and integration with real-time wellness interventions. As wearable technologies continue to advance, machine-learning-driven stress detection has the potential to support healthier and more adaptive workplace environments.

---

## References
1.    Richer, R., Koch, V., Abel, L., Hauck, F., Kurz, M., Ringgold, V., Müller, V., Küderle, A., Schindler-Gmelch, L., Eskofier, B. M., & Rohleder, N. (2024). Machine learning-based detection of acute psychosocial stress from body posture and movements. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-59043-1
2.    Sabry, F., Eltaras, T., Labda, W., Alzoubi, K., & Malluhi, Q. (2022). Machine Learning for Healthcare Wearable Devices: The big picture. Journal of Healthcare Engineering, 2022, 1–25. https://doi.org/10.1155/2022/4653923
3.    Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction, 400–408. https://doi.org/10.1145/3242969.3242985
4.    Razavi, M., Ziyadidegan, S., Mahmoudzadeh, A., Kazeminasab, S., Baharlouei, E., Janfaza, V., Jahromi, R., & Sasangohar, F. (2024). Machine Learning, deep learning, and data preprocessing techniques for detecting, predicting, and monitoring stress and stress-related mental disorders: Scoping review. JMIR Mental Health, 11. https://doi.org/10.2196/53714
5.    Strudwick, J., Gayed, A., Deady, M., et al. (2023). Workplace mental health screening: A systematic review and meta-analysis. Occupational and Environmental Medicine, 80(8), 469–484. https://doi.org/10.1136/oemed-2022-108608
6.    Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, Ł., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng, X. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. https://www.tensorflow.org
7.    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org/
