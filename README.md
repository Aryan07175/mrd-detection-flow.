MRD Detection in Flow Cytometry using VAE and GMM
This repository presents two complementary approaches for detecting Minimal Residual Disease (MRD) in flow cytometry data:

VAE/: Deep learning-based anomaly detection using Variational Autoencoders
GMM/: Probabilistic modeling with Gaussian Mixture Models
Dataset Overview
We work with flow cytometry data collected from 12 patients:

Healthy Patients (P1–P6):
~27 million cells
Used for model training
Unhealthy Patients (P7–P12):
~20 million cells
Used for evaluation and MRD prediction
Each cell is represented by 14 features. The models are trained to learn the healthy distribution and detect anomalous cells in new patient data, which may indicate MRD.

Objective
To accurately identify cells that are anomalous (i.e., likely cancerous) using only unsupervised learning methods trained on healthy patient data. These anomalies collectively form an estimate of MRD (%).

Methods Used
1. Variational Autoencoder (VAE)
Learns latent representations via probabilistic encoding/decoding
Detects anomalies based on reconstruction error (MSE)
Explored different latent dimensions (2 and 4) and β values
Uses Leave-One-Patient-Out (LOPO) validation with progressive fine-tuning
Produces per-cell MSE scores → used to estimate MRD
📎 Explore VAE Approach
📎 VAE Best Model

2. Gaussian Mixture Model (GMM)
Trains a mixture of Gaussians on healthy cell data
Evaluates likelihood of each new cell under the model
Low-likelihood cells are flagged as anomalies
Tried multiple component counts (4, 6, 16)
Compared full vs tied covariance structures
Final threshold: 1.5th percentile of healthy scores
📎 Explore GMM Approach
📎 GMM Best Model

Evaluation Metrics
Mean Squared Error (MSE) between predicted and actual MRD %
Mean Absolute Error (MAE)
MRD Estimation for each patient based on anomaly scores
Both models approximate expert-annotated MRD scores with high accuracy.

Main libraries:
torch
scikit-learn
numpy, pandas
matplotlib, seaborn
joblib
References
PyTorch VAE Tutorial
Uncovering Anomalies with Variational Autoencoders – Towards Data Science
Hands-On Anomaly Detection with Variational Autoencoders – Medium
scikit-learn GMM: https://scikit-learn.org/stable/modules/mixture.html
Understanding Gaussian Mixture Models – Number Analytics Blog
PMC Article on GMM & MRD
About
Unsupervised MRD detection in flow cytometry data using Variational AutoEncoder (VAE) and Gaussian Mixture Model (GMM).

Topics
sklearn pytorch gaussian-mixture-models unsupervised-learning gmm flow-cytometry anomaly-detection variational-autoencoder cancer-detection mrd vae-implementation vae-pytorch
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Languages
Jupyter Notebook
98.8%
 
HTML
1.2%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Sec
