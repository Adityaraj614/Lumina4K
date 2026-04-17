import torch
import os

# This uses PyTorch official hub (reliable)
model = torch.hub.load(
    'pytorch/examples',
    'fast_neural_style',
    model='mosaic'
)

save_path = "weights/styles/mosaic.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save model weights
torch.save(model.state_dict(), save_path)

print("✅ Model downloaded and saved at:", save_path)