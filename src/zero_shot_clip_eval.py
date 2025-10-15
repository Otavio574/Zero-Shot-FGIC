# zero_shot_clip_eval.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ===== Configurações =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16  # ajuste se memória não for suficiente

CSV_PATH = "stanford_dogs_labels.csv"  # CSV gerado pelo prepare_stanford_dogs.py

# ===== Carregando modelo =====
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE) #type: ignore
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# ===== Carregando dataset =====
df = pd.read_csv(CSV_PATH)
labels = df['label'].unique().tolist()

# Criando prompts Zero-Shot
prompts = [f"a photo of a {label}" for label in labels]

# Calculando embeddings dos textos
with torch.no_grad():
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE) #type: ignore
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

# ===== Função para processar imagens por batch =====
def batch(iterable, n=1):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i + n]

# ===== Avaliação =====
top1_correct = 0
top5_correct = 0
total = len(df)

for img_paths in batch(df['image_path'].tolist(), BATCH_SIZE):
    images = [Image.open(p).convert("RGB") for p in img_paths]
    with torch.no_grad():
        img_inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE) #type: ignore
        img_embeddings = model.get_image_features(**img_inputs)
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        # Similaridade coseno
        logits = img_embeddings @ text_embeddings.T
        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices

    # Comparando com labels reais
    for idx, img_idx in enumerate(img_paths):
        true_label = df.iloc[df.index[df['image_path'] == img_idx][0]]['label']
        true_idx = labels.index(true_label)
        if top1[idx].item() == true_idx:
            top1_correct += 1
        if true_idx in top5[idx].tolist():
            top5_correct += 1

# ===== Resultados =====
top1_acc = top1_correct / total
top5_acc = top5_correct / total
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
