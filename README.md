# COVID-19 detection from coughs

 **Note:** This repository is still in progress.  
The current codebase includes fully implemented and well-documented pipelines for audio preprocessing, feature extraction (MFCC, MFSC, Mel spectrograms), dataset preparation, and optimized DTW-based similarity computation.  
The training scripts for HuBERT/AST/VGG models will be added once the code is polished.

Use this [**Hugging Face dataset**](https://huggingface.co/datasets/marceltomas/covid-cough-detection) to reproduce the experiments, which is a balanced selection of recordings from 3 different databases: [**Coswara**](https://arxiv.org/abs/2005.10548), [**Cambridge**](https://www.covid-19-sounds.org/es/blog/voice_covid_icassp.html), and [**Coughvid**](https://zenodo.org/records/4048312).

## ✅ Implemented
- Modular audio preprocessing pipeline.
- Feature extraction: MFCC, MFSC, Mel spectrograms.
- Efficient PyTorch + NumPy implementation of feature pipelines.
- Custom Hugging Face dataset.
- DTW-based similarity engine (KNN + optimized vectorized DTW).

## 🚧 In Progress
- Training scripts for HuBERT, AST, and VGG models
- Hyperparameter sweep utilities.
- Final evaluation and documentation.

All preprocessing, feature extraction, and DTW computations were implemented from scratch.
This repository is built from this **[Kaggle competition](https://www.kaggle.com/competitions/covid4)** used in one of my university courses.
