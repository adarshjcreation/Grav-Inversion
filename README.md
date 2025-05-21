# Grav-Inversion
Gravity inversion using the Reg-UNet network

# 📂 Structure
```
📦 Grav-Inversion/
├── 📁 data/                 # Dataset storage  
├── 📁 models/               # Trained model checkpoints    
├── 🚀 train.py              # Main training script  
├── 🏗️ model.py              # Reg-Unet model architecture  
├── 📜 dataset.py            # Dataset loading & DataLoader  
├── 📉 loss.py               # Loss functions (Dice loss, etc.)  
├── ⚙️ config.py             # Hyperparameters & configuration  
├── 📄 requirements.txt       # List of dependencies  
├── 📜 README.md              # Project documentation  
```
## 📊 Model Architecture  
Here is a visualization of our **Reg-UNet model** used for gravity inversion:

![Reg-UNet Model Architecture](assets/model_arc.png)



# References:
1. Huang, R., Liu, S., Qi, R., & Zhang, Y. (2021). Deep learning 3D sparse inversion of gravity data. Journal of Geophysical Research: Solid Earth, 126(11), e2021JB022476.

