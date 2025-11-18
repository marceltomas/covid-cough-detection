# **COVID-19 Detection from Cough & Breath Audio**

This repository contains an end-to-end deep learning pipeline for COVID-19 detection using cough and breath recordings. It includes preprocessing, feature extraction, model training (CNN and Transformer-based), and evaluation tools.

The project uses the following [**Hugging Face dataset**](https://huggingface.co/datasets/marceltomas/covid-cough-detection) to reproduce the experiments, which is a balanced selection of recordings from 3 different databases: [**Coswara**](https://arxiv.org/abs/2005.10548), [**Cambridge**](https://www.covid-19-sounds.org/es/blog/voice_covid_icassp.html), and [**Coughvid**](https://zenodo.org/records/4048312).

## 🗂️ Project Structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis + feature extraction.
│   └── 02_model_analysis.ipynb   # VGG experiments and evaluation.
│
├── config.py                     # Constants and hyperparameters.
├── data_preprocessing.py         # Preprocessing the Kaldi-style dataset.
├── feature_extraction.py         # MFCC, MFSC, MelSpectrogram implementations.
├── models.py                     # VGG, HuBERT, AST architecture code.
├── training.py                   # Training loop, evaluation.
├── utils.py                      # Misc utilities (DTW, kNN...).
├── requirements.txt
└── README.md
```
---

### **Processing**

* Unified preprocessing.
* Predict missing audio types using DTW-based kNN.
* MFCC, MFSC, and Mel Spectrogram extraction (NumPy + PyTorch).

### **HuggingFace Dataset**

* Kaldi-style directory layout. 
* Standard train/test splits with .wav files.
* Precomputed DTW distance lookup.
* Precomputed MFSC, MFCC and Mel Spectrograms for each audio file.

### **Models**

* **VGG** for audio classification.
* **HuBERT** training pipeline.
* **AST (Audio Spectrogram Transformer)** training pipeline.
* Flexible feature backends (MFCC, MFSC, MelSpec, raw waveform).

### **Training & Evaluation**

* Early stopping + learning-rate scheduling.
* Metrics: AUC, accuracy, precision, recall.
* Cross-domain evaluation analysis.

## 📝 Notebooks

Two notebooks provide a deep walkthrough:

* **01_eda.ipynb** — EDA, preprocessing, MFCC/MFSC/Mel explanations.
* **02_model_analysis.ipynb** — VGG, HuBERT, and AST experiments, evaluation metrics.

These demonstrate the full ML workflow used during development.

---

The objective is to provide a **modular, reproducible baseline** for audio-based diagnosis systems, enabling:

* Benchmarking CNN and Transformer architectures.
* Exploration of audio feature pipelines.
* Testing cross-domain generalization.

This repository is built from this **[Kaggle competition](https://www.kaggle.com/competitions/covid4)** used in one of my university courses.

