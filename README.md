# COVID-19 detection from coughs

 **Note:** This repository is still in progress.  
The current codebase includes fully implemented and well-documented pipelines for audio preprocessing, feature extraction (MFCC, MFSC, Mel spectrograms), dataset preparation, and optimized DTW-based similarity computation.  
The training scripts for HuBERT/AST/VGG models will be added once the code is polished.

Use this [Hugging Face dataset](https://huggingface.co/datasets/marceltomas/covid-cough-detection) to reproduce the experiments.

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
