import torch

# Training Configuration
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
PATIENCE = 20
THRESHOLD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths
DATA_PATH = "data/data{}.mat"
SYN_DATA_PATH = "data/syn1_data/data{}.mat"
MODEL_SAVE_PATH = "models/reg_unet.pth"
