import torch
from transformer_net import TransformerNet  # from repo

# Load model
model = TransformerNet()
model.load_state_dict(torch.load("weights/styles/mosaic.pth"))
model.eval()

# Convert to TorchScript
example_input = torch.rand(1, 3, 512, 512)

traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("weights/styles/anime.pt")

print("✅ Model converted to anime.pt")