from braindecode.models import EEGConformer
import mne
import numpy as np
import torch

from skorch.dataset import ValidSplit
from braindecode import EEGClassifier

N_CHANNELS = 4
N_OUTPUTS = 5
N_TIMES = 1024


# Create sample data
info = mne.create_info(ch_names=["C3", "C4", "C5", "Cz"], sfreq=256.0, ch_types="eeg")
X = np.random.randn(
    100, N_CHANNELS, N_TIMES
)  # 100 epochs, 4 channels, 4 seconds (@256Hz)
epochs = mne.EpochsArray(X, info=info)
y = np.random.randint(0, N_OUTPUTS, size=100)  # 5 classes

# Initialize model with hyperparameters
model = EEGClassifier(
    module=EEGConformer,
    module__n_chans=N_CHANNELS,
    module__n_outputs=N_OUTPUTS,
    module__n_times=N_TIMES,
    module__final_fc_length="auto",
    # Training hyperparameters
    max_epochs=30,
    batch_size=32,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.001,
    optimizer__weight_decay=0.01,
    # Use 20% of training data as validation set
    train_split=ValidSplit(0.2),
    # Enable GPU if available
    device="cuda" if torch.cuda.is_available() else "cpu",
    # Progress tracking
    iterator_train__shuffle=True,
    verbose=1,
)
model.fit(epochs, y)