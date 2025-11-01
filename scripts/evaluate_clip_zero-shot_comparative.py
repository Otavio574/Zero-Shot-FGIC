"""
Avaliação Zero-Shot COMPARATIVA com CLIP.
Compara pares de classes ("mais parecido com A do que com B").
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from pathlib import Path

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega datasets do summary.json"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    datasets = {}
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    return datasets


SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_comparative"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """Carrega descrições (ou cria genéricas)"""
    desc_path = os.path.join("descriptors", f"{dataset_name}_descriptors.json")
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️  Nenhum descriptor encontrado para {dataset_name}, usando genéricos.")
        return {}


def load_embeddings(dataset_name):
    """Carrega embeddings de imagem"""
    emb_path = os.path.join("embeddings", f"{dataset_name}.pt")
    if not os.path.exists(emb_path):
        print(f"❌ Embeddings não encontrados: {emb_path}")
        return None, None, None
    data = torch.load(emb_path, weights_only=False)
    if isinstance(data, dict):
        return data["image_embeddings"], data.get("image_paths", None), data.get("labels", None)
    return data, None, None


def infer_classes_from_paths(image_paths):
    """Extrai nomes de classes dos caminhos"""
    class_names = []
    labels = []
    class_to_idx = {}

    for path in image_paths:
        parts = Path(path).parts
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_names)
            class_names.append(class_name)
        labels.append(class_to_idx[class_name])

    return np.array(labels), class_names


def evaluate_comparative(dataset_name, image_embeds, image_paths, descriptions, model):
    """Avaliação comparativa por pares"""

    print(f"\n🔍 Iniciando avaliação comparativa: {dataset_name}")
    labels, class_names = infer_classes_from_paths(image_paths)
    num_classes = len(class_names)

    print(f"   Classes detectadas: {num_classes}")
    print(f"   Total de imagens: {len(image_embeds)}")

    accuracies = []

    # Pré-computar frases baseadas nas classes
    class_texts = {}
    for cls in class_names:
        if descriptions and cls in descriptions:
            base = descriptions[cls]
        else:
            base = f"a photo of a {cls.replace('_', ' ')}"
        class_texts[cls] = base

    for i, image_embed in enumerate(tqdm(image_embeds, desc=f"Avaliando {dataset_name}")):
        true_idx = labels[i]
        true_class = class_names[true_idx]

        correct = 0
        total = 0

        for j, other_class in enumerate(class_names):
            if other_class == true_class:
                continue

            text_pos = f"This image looks more like a {true_class} than a {other_class}."
            text_neg = f"This image looks more like a {other_class} than a {true_class}."

            tokens = clip.tokenize([text_pos, text_neg]).to(DEVICE)

            with torch.no_grad():
                text_features = model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                sim = (image_embed.to(DEVICE).to(torch.float32) @ text_features.to(torch.float32).T).squeeze(0)

                if sim[0] > sim[1]:
                    correct += 1
                total += 1

        if total > 0:
            accuracies.append(correct / total)

    overall_acc = np.mean(accuracies)
    print(f"✅ Acurácia comparativa: {overall_acc:.4f}")
    return overall_acc, len(class_names), len(image_embeds)


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot COMPARATIVA com CLIP ({MODEL_NAME})")
    print(f"💻 Device: {DEVICE}\n")

    model, _ = clip.load(MODEL_NAME, device=DEVICE)

    summary = {
        "model": MODEL_NAME,
        "method": "comparative",
        "total_datasets": len(DATASETS),
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"{'='*60}")

        image_embeds, image_paths, _ = load_embeddings(dataset_name)
        if image_embeds is None:
            print(f"⚠️  Pulando {dataset_name} (sem embeddings)")
            continue

        descriptions = load_descriptions(dataset_name)

        acc, num_classes, num_images = evaluate_comparative(
            dataset_name, image_embeds, image_paths, descriptions, model
        )

        summary["results"][dataset_name] = {
            "accuracy": float(acc),
            "num_classes": num_classes,
            "num_images": num_images
        }

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_comparative.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ Avaliação finalizada com sucesso.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
