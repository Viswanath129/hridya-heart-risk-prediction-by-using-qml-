# Research Note: Hybrid Quantum-Inspired Heart Disease Risk Assessment (HRIDYA)

## Abstract
This report evaluates the application of Variational Quantum Circuits (VQC) in the domain of cardiovascular risk assessment. By comparing a quantum-inspired model (HRIDYA-QML) against established classical baselines, we investigate the efficacy of hybrid architectures in processing clinical data.

## Introduction
Heart disease remains a leading cause of global mortality. While classical machine learning has made significant strides, quantum-inspired methods offer a new paradigm for modeling non-linear feature interactions. This study presents HRIDYA, a hybrid framework designed to explore these capabilities.

## Methodology
The dataset consists of 500 clinical samples with features including Age, Blood Pressure, Cholesterol, and a simulated QuantumPatternFeature.
- **Classical Pipeline**: Logistic Regression, Random Forest, and Support Vector Machines (SVM).
- **Quantum Pipeline**: A 6-qubit circuit utilizing Angle Embedding for feature mapping and Strongly Entangling Layers for variational optimization using the Adam optimizer.

## Experiments
The models were trained and evaluated on a 20% test split. 
- Classical models achieved high accuracy and precision, serving as a robust benchmark.
- HRIDYA-QML achieved a lower overall accuracy but a significantly higher recall (0.967), indicating a low false-negative rate in identifying risk.

## Discussion
The high recall of the QML model suggests that quantum-inspired feature maps might be effectively capturing "tail-end" risk indicators that linear models might marginalize. However, the simulation complexity and current hardware noise limitations remaining significant barriers to immediate clinical deployment.

## Conclusion
HRIDYA demonstrates that quantum-inspired models can provide competitive, high-sensitivity performance in heart disease prediction. Future work will focus on error-mitigation techniques and validation on larger, multi-institutional datasets.

---
*Date: February 2026*
*Lead Researcher: HRIDYA Team*
