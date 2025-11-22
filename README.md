# MRD Detection in Flow Cytometry using VAE and GMM

This project presents two complementary unsupervised machine learning approaches for detecting Minimal Residual Disease (MRD) in flow cytometry data:

- Variational Autoencoder (VAE)
- Gaussian Mixture Model (GMM)

---

## Project Overview

The goal of this project is to estimate MRD (%) in patients by detecting anomalous (cancer-like) cells using models trained only on healthy patient data.  
No labels are used — making the approach fully unsupervised.

---

## Dataset Overview

Flow cytometry data was collected from 12 patients:

### Healthy Patients (P1–P6)
- ~27 million cells  
- Used for model training

### Unhealthy Patients (P7–P12)
- ~20 million cells  
- Used for evaluation and MRD prediction

Each cell contains 14 numerical features representing fluorescence and scatter properties.  
The models learn the distribution of healthy cells and flag those that deviate as anomalies.

---

## Objective

To accurately detect MRD using anomaly scores produced by:
- VAE reconstruction error
- GMM likelihood scores

Anomalous cells are aggregated to estimate MRD (%).

---

# Methods Used

## 1. Variational Autoencoder (VAE)

A deep learning model that learns latent representations of healthy cells.

### Key Features:
- Probabilistic encoding and decoding
- Anomaly detection using reconstruction error (MSE)
- Tested latent dimensions = 2 and 4
- Used β-VAE variations
- LOPO (Leave-One-Patient-Out) validation
- Progressive fine-tuning
- Output: per-cell MSE scores → MRD estimation

Explore VAE Approach  
VAE Best Model

---

## 2. Gaussian Mixture Model (GMM)

A probabilistic model used to identify anomalous cells.

### Key Features:
- Gaussian mixtures trained on healthy data
- Cells evaluated using log-likelihood scores
- Low-likelihood cells flagged as anomalies
- Tested component counts: 4, 6, 16
- Compared full and tied covariance structures
- Final anomaly threshold: 1.5th percentile of healthy scores
- Output: anomaly detection → MRD estimation

Explore GMM Approach  
GMM Best Model

---

# Evaluation Metrics

Both models were evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Correlation with expert-annotated MRD values

Both approaches approximate expert MRD scores with high accuracy.

---

# Libraries Used

- torch  
- scikit-learn  
- numpy  
- pandas  
- matplotlib  
- seaborn  
- joblib  

---

# References

- PyTorch VAE Tutorial  
- Uncovering Anomalies with Variational Autoencoders – Towards Data Science  
- Hands-On Anomaly Detection with Variational Autoencoders – Medium  
- scikit-learn GMM Documentation  
- Understanding Gaussian Mixture Models – Number Analytics  
- PMC Article on GMM & MRD  

---

# About

Unsupervised MRD detection in flow cytometry data using:
- Variational AutoEncoder (VAE)
- Gaussian Mixture Model (GMM)

Designed for medical anomaly detection, cancer diagnostics, and biomedical research.

---

# Topics

sklearn, pytorch, gaussian-mixture-models, unsupervised-learning, gmm,  
flow-cytometry, anomaly-detection, variational-autoencoder, cancer-detection,  
mrd, vae-implementation, vae-pytorch

