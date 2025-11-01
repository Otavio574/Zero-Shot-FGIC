import torch

data = torch.load("embeddings/Stanford_Dogs.pt")
print("Keys:", data.keys())
print("Embeddings shape:", data["image_embeddings"].shape)
print("First image path:", data["image_paths"][0])