# 3. Related Work

Anomaly detection in time-series data has been extensively studied, with approaches broadly categorized into statistical, machine learning, and deep learning methods. Traditional machine learning methods like Isolation Forest (IF) and One-Class SVM are widely used for their efficiency and robustness to outliers in high-dimensional feature spaces. These methods typically rely on statistical features (e.g., RMS, kurtosis, skewness) extracted from raw signals. However, their performance often degrades when underlying data distributions shift due to changing operational conditions.

Deep learning methods, particularly Autoencoders (AE) and their variants like Convolutional Autoencoders (CNN-AE), have shown superior capability in capturing complex, non-linear dependencies in raw time-series data without manual feature engineering. While AEs are powerful for modeling normal patterns, maintaining their robustness under condition shifts (e.g., varying loads in rotating machinery) remains a critical challenge. 

Furthermore, the selection of anomaly thresholds significantly impacts model performance. Static thresholds based on fixed percentiles or standard deviations ($\mu + 3\sigma$) often fail when score distributions drift. Dynamic thresholding techniques, such as Peak-Over-Threshold (POT) based on Extreme Value Theory, adapt better to varying distributions but require careful parameter tuning. In this study, we aim to bridge the gap by systematically evaluating the robustness of both classical (IF) and deep learning (CNN-AE) approaches under condition shifts and varying noise levels, combined with an ablation on thresholding strategies.

# 5. Methods

## Baseline + Comparison

As a baseline for comparison, we employ the Isolation Forest (IF) algorithm, a popular tree-based ensemble method for anomaly detection. Unlike methods that profile normal data, IF explicitly isolates anomalies by randomly partitioning the feature space, exploiting the fact that anomalies are "few and different" and thus require fewer splits to be isolated.

For the IF model, we extract eight statistical features from each time-series window: Root Mean Square (RMS), standard deviation, mean absolute value, peak value, peak-to-peak amplitude, crest factor, skewness, and kurtosis. These features are commonly used in machinery fault diagnosis to capture the signal's energy and distribution characteristics. The IF model is configured with `n_estimators=200` to ensure stable predictions. The anomaly score is defined as the negative of the sample's average path length in the forest, such that higher scores indicate greater abnormality.

To comprehensively evaluate the models, we employ several thresholding strategies:
1. **Percentile 99**: A fixed empirical threshold assuming 1% of validation data are anomalies.
2. **Mean + 3 Standard Deviations ($\mu + 3\sigma$)**: A statistical threshold based on the distribution of normal scores.
3. **Peak-Over-Threshold (POT)**: A dynamic thresholding method that models the tail of the score distribution using the Generalized Pareto Distribution.
4. **Condition-Specific Thresholding**: Thresholds calibrated separately for each operational load using a small calibration set.

Our comparison strategy focuses on contrasting the feature-engineered IF baseline against the representation-learning CNN-AE across different loads (Setup 1 & 2) and noise levels (Setup 3) to evaluate their robustness against condition shifts.

# 10. References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413-422). IEEE.
2. Siffer, A., Fouque, P. A., Termier, A., & Largouet, C. (2017). Anomaly detection in streams with extreme value theory. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1067-1075).
3. Borghesi, A., Bartolini, A., Lombardi, M., Milano, M., & Benini, L. (2019). Anomaly detection using autoencoders in high performance computing systems. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, No. 01, pp. 9428-9433).
4. Zhao, R., Yan, R., Chen, Z., Mao, K., Wang, P., & Gao, R. X. (2019). Deep learning and its applications to machine health monitoring. *Mechanical Systems and Signal Processing*, 115, 213-237.
5. Hundt, L., et al. (2020). Condition monitoring of rotating machinery under varying operating conditions. *IEEE Transactions on Industrial Electronics*, 68(9), 8754-8763.
6. Chalapathy, R., & Chawla, S. (2019). Deep learning for anomaly detection: A survey. *arXiv preprint arXiv:1901.03407*.
7. Li, X., Zhang, W., Ding, Q., & Sun, J. (2020). Intelligent rotating machinery fault diagnosis based on deep learning using data augmentation. *Journal of Intelligent Manufacturing*, 31(2), 433-452.
8. Ruff, L., et al. (2021). A unifying review of deep and shallow anomaly detection. *Proceedings of the IEEE*, 109(5), 756-795.
9. Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2828-2837).
10. Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. *Mechanical Systems and Signal Processing*, 64, 100-131.
