import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import regUnet
from dataset import CustomDataset, load_data
from loss import my_loss
from config import *
import wandb

wandb.init(project="Reg-Unet-Training")

# Load Data
tra_x, tra_y = load_data(DATA_PATH, 20000)
val_x, val_y = load_data(DATA_PATH, 2000)

# Create Dataloaders
tra_loader = DataLoader(CustomDataset(tra_x, tra_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CustomDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False)

# Model, Optimizer, and Scheduler
model = regUnet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Training Loop
best_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in tra_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = my_loss(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            val_loss += my_loss(pred, y).item()

    val_loss /= len(val_loader)
    train_loss /= len(tra_loader)
    scheduler.step()

    # Log Results
    wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Validation Loss": val_loss})

    # Save Best Model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

wandb.finish()
