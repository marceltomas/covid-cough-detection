import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn               
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from typing import Optional, Tuple  
import copy                        
import time
import math
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor

AUDIO_TYPE_MAPPING = {
    'cough': 0,
    'breath': 1,
}

class WaveformDataset(Dataset):
    """Custom Dataset from a DataFrame for raw waveform models."""
    def __init__(self, df, feature_col="raw_waveform", label_col="label_id", audio_type_col="audio_type"):
        self.wavs = df[feature_col].values   # Shape (1,T) or (T,)
        self.labels = df[label_col].values
        self.audio_types = df[audio_type_col].map(AUDIO_TYPE_MAPPING).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        if wav.ndim == 2: 
            wav = wav.squeeze(0)            # Shape (1,T) -> (T,)
        return wav.astype(np.float32), int(self.labels[idx]), self.audio_types[idx]

class FeatureDataset(Dataset):
    """
    Custom Dataset to load features and labels from a DataFrame (for spectrogram models).
    Each sample is padded or truncated to a fixed number of frames,
    where max_len is automatically inferred from the 90th percentile
    of sample lengths (unless provided manually).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_col: str,
        label_col: str = "label_id",
        audio_type_col="audio_type",
        max_len: Optional[int] = None,
        percentile: int = 90,
    ):
        """
        Args:
            df: DataFrame containing feature and label columns.
            feature_col: Name of the column with feature arrays (shape (n_feats, n_frames)).
            label_col: Name of the column with labels. Defaults to 'label_id'.
            max_len: Optional fixed number. If None, it will be computed as the given percentile of frame lengths.
            percentile: Percentile used to compute max_len when max_len=None.
        """
        self.features = []
        self.labels = []
        self.audio_types = df[audio_type_col].map(AUDIO_TYPE_MAPPING).values
        if max_len is None:
            n_frames_list = [feat.shape[1] for feat in df[feature_col]]
            max_len = int(np.percentile(n_frames_list, percentile))
            print(f"[FeatureDataset] Using max_len={max_len} (p{percentile})")

        self.max_len = max_len
        for _, row in df.iterrows():
            feature = row[feature_col]  # shape (n_feats, n_frames)
            label = row[label_col]

            if feature.shape[1] < max_len:
                pad_width = max_len - feature.shape[1]
                pad = np.zeros((feature.shape[0], pad_width), dtype=np.float32)
                feature = np.hstack((pad, feature))  # pad at beginning
            elif feature.shape[1] > max_len:
                feature = feature[:, -max_len:]      # keep last max_len frames

            self.features.append(feature)
            self.labels.append(label)

        self.features = torch.tensor(np.stack(self.features), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.audio_types = torch.tensor(self.audio_types, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.audio_types[idx]
    
def train_epoch(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    log_interval: int = 10,
    use_audio_type: bool = False,
    verbose: bool = True,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch_idx, (data, target, audio_type) in enumerate(loader):
        data, target, audio_type = data.to(device), target.to(device), audio_type.to(device)
        optimizer.zero_grad()
        if use_audio_type:
            output = model(data, audio_type)
        else:
            output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

        if verbose and batch_idx % log_interval == 0:
            avg_loss = running_loss / total_samples
            print(f"Epoch {epoch:02d} [{total_samples:5d}/{len(loader.dataset)} "
                  f"({100 * total_samples / len(loader.dataset):5.1f}%)] "
                  f"Loss: {avg_loss:.6f}")

    return running_loss / total_samples

def roc_auc_score_ci(y_true, y_score, positive=1):
    """
    95% Confidence Interval for AUC. Hanley and McNeil (1982).
    https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
    """
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = math.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return AUC, (lower, upper)

def evaluate_model(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    use_audio_type: bool = False,
    verbose: bool = True,
    dataset_name: str = "Validation",
) -> Optional[tuple[float, float]]:
    """
    Evaluate the model on a labeled dataset.
    Returns (avg_loss, auc) if labels exist, otherwise (None, None)
    """
    sample_batch = next(iter(loader))
    _, target_sample, _ = sample_batch
    if (target_sample < 0).all():
        if verbose:
            print(f"\n{dataset_name} set has no labels — skipping evaluation.\n")
        return None, None

    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for data, target, audio_type in loader:
            data, target, audio_type = data.to(device), target.to(device), audio_type.to(device)
            if use_audio_type:
                output = model(data, audio_type)
            else:
                output = model(data)
            pred = output.sigmoid()

            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

            loss = criterion(output, target.float()).item()
            total_loss += loss * data.size(0)

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    if len(np.unique(y_true)) == 2:
        auc, (ci_low, ci_high) = roc_auc_score_ci(y_true, y_pred)
    else:
        auc = ci_low = ci_high = float("nan")

    if verbose:
        print(
            f"\n{dataset_name} set: Average loss: {avg_loss:.4f}, "
            f"AUC: {100 * auc:.2f}% "
            f"({100 * ci_low:.2f}% - {100 * ci_high:.2f}%)\n"
        )

    return avg_loss, auc

def collate_fn(batch):
    wavs, labels, audio_types = zip(*batch)
    inputs = collate_fn.processor(
        list(wavs),
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    return inputs["input_values"], torch.tensor(labels), torch.tensor(audio_types, dtype=torch.long)

def get_dataloader(
    df: pd.DataFrame, 
    feature_col: str, 
    max_len: Optional[int] = None, 
    percentile: int = 90,
    batch_size: int = 32, 
    processor_name: Optional[str] = None, 
    shuffle: bool =True,
    label_col: str ="label_id", 

):
    """
    Returns DataLoader for the given dataframe based on the feature column.
    Supports waveform and feature-based datasets.
    """
    if feature_col == "raw_waveform":
        dataset = WaveformDataset(df, feature_col=feature_col)
        try: # Attach processor from AutoProcessor
            collate_fn.processor = AutoProcessor.from_pretrained(processor_name)
        except Exception: # Fallback for models with no tokenizer (HuBERT, WavLM)
            collate_fn.processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    else:
        dataset = FeatureDataset(df, feature_col=feature_col, max_len=max_len, percentile=percentile)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    feature_col: str,
    device: torch.device,
    max_len: Optional[int] = None,
    percentile: int = 90,
    batch_size: int = 32,
    optimizer: str = "adam",
    lr: float = 0.0001,
    momentum: float = 0.9,
    epochs: int = 100,
    patience: int = 5,
    log_interval: int = 5,
    use_audio_type: bool = False,
    verbose: bool = True,
    processor_name: str = None,
):
    """Train a model with early stopping based on validation AUC."""
    train_loader = get_dataloader(train_df, feature_col, max_len, percentile, batch_size, processor_name, shuffle = True)
    test_loader = get_dataloader(test_df, feature_col, max_len, percentile, batch_size, processor_name, shuffle = False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer.lower() == "adam" else optim.SGD(model.parameters(), lr=lr, momentum=momentum)  
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    model.to(device)
    best_valid_auc = 0.0
    best_model_state = None
    state = None
    iteration = 0
    epoch = 1
    t0 = time.time()
    
    while (epoch < epochs + 1) and (iteration < patience):
        train_epoch(train_loader, model, criterion, optimizer, epoch, device, log_interval, use_audio_type, verbose)
        valid_loss, valid_auc = evaluate_model(test_loader, model, criterion, device, use_audio_type, verbose, dataset_name='Validation')
        scheduler.step(valid_auc)    
        if valid_auc <= best_valid_auc:
            iteration += 1
            if verbose:
                print('AUC was not improved, iteration {0}'.format(str(iteration)))
        else:
            iteration = 0
            best_valid_auc = valid_auc
            best_model_state = copy.deepcopy(model.state_dict())
            state = {
                'valid_auc': best_valid_auc,
                'valid_loss': valid_loss,
                'epoch': epoch,
                'time': time.time() - t0,
            }
        epoch += 1
        if verbose:
            print(f'Elapsed seconds: ({time.time() - t0:.0f}s)')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, state 