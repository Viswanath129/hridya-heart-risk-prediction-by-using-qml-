# HRIDYA: Hybrid Quantum-Inspired Heart Disease Risk Assessment

### Overview
HRIDYA evaluates hybrid quantum-inspired machine learning for heart disease risk prediction by comparing classical ML models with a variational quantum classifier on structured clinical data.

### Research Question
Can quantum-inspired models capture non-linear physiological interactions beyond classical baselines?

### Methodology
- **Classical models**: Logistic Regression, Random Forest, SVM
- **Quantum model**: Variational Quantum Circuit (VQC) with Angle Encoding and Strongly Entangling Layers.
- **Metrics**: Accuracy, Precision, Recall, F1-Score.

### Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| LogisticRegression | 0.950 | 1.000 | 0.917 | 0.957 |
| RandomForest | 0.930 | 1.000 | 0.883 | 0.938 |
| SVM | 0.930 | 0.965 | 0.917 | 0.940 |
| **HRIDYA-QML** | 0.730 | 0.699 | 0.967 | 0.811 |

### Observations
- **Classical Dominance**: Traditional models show high precision and overall accuracy on the structured dataset.
- **QML Sensitivity**: HRIDYA-QML demonstrates higher recall (0.967), suggesting it may be more sensitive to a broader range of risk patterns, though at the cost of higher false positives in this simulation.
- **Non-linear Modeling**: The VQC approach highlights potential for modeling complex interactions through quantum entanglement.

### Limitations
- **Simulated Execution**: Work performed on classical simulators (default.qubit).
- **Dataset Scale**: Evaluation on a constrained clinical dataset.

### Future Work
- Deployment on actual Quantum Hardware (e.g., IBM Q, Rigetti).
- Integration of longitudinal patient history.
- Refinement of variational circuit depth.

---
*Developed as a research-grade artifact for Hybrid ML exploration.*
