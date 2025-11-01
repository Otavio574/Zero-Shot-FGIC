"""
Avaliação Zero-Shot com CLIP usando Comparative Filtering.
Usa embeddings pré-calculados e filtra classes muito similares.
Método: Remove classes cujos text embeddings são muito similares entre si.
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega configuração de datasets do summary.json"""
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

MODEL_NAME = "ViT-B/32"  # Modelo CLIP do OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_filtering"

# Parâmetro de filtragem
SIMILARITY_THRESHOLD = 0.7  # Remove classes com similaridade média > 0.7

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """Carrega descriptions do dataset"""
    path = os.path.join("descriptors", f"{dataset_name}_descriptors.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️  Nenhum descriptor encontrado para {dataset_name}, usando genéricos.")
        return {}


def load_embeddings(dataset_name):
    """Carrega embeddings de imagem"""
    emb_path = os.path.join("embeddings", f"{dataset_name}.pt")
    if not os.path.exists(emb_path):
        print(f"❌ Embeddings não encontrados: {emb_path}")
        return None, None
    
    data = torch.load(emb_path, weights_only=False)
    if isinstance(data, dict):
        return data["image_embeddings"], data.get("image_paths", None)
    return data, None


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


def apply_comparative_filtering(text_embeds, class_names, threshold=0.7):
    """
    Aplica filtro comparativo: remove classes muito similares entre si.
    
    Args:
        text_embeds: embeddings de texto [num_classes, dim]
        class_names: lista de nomes das classes
        threshold: limiar de similaridade média
    
    Returns:
        text_embeds filtrados, class_names filtrados, índices mantidos
    """
    print(f"🔍 Aplicando filtro comparativo (threshold={threshold})...")
    
    keep_indices = []
    for i in range(len(text_embeds)):
        sims = (text_embeds[i] @ text_embeds.T).detach().cpu().numpy()
        # Penaliza similaridade excessiva (exclui auto-similaridade)
        avg_sim = (np.sum(sims) - 1.0) / (len(sims) - 1)
        
        if avg_sim < threshold:
            keep_indices.append(i)
    
    filtered_text_embeds = text_embeds[keep_indices]
    filtered_class_names = [class_names[i] for i in keep_indices]
    
    print(f"✅ {len(filtered_class_names)}/{len(class_names)} classes mantidas após filtragem")
    
    return filtered_text_embeds, filtered_class_names, keep_indices


def evaluate_with_filtering(dataset_name, image_embeds, image_paths, descriptions, model):
    """Avaliação com filtering de classes similares"""
    
    print(f"\n🔍 Iniciando avaliação com filtering: {dataset_name}")
    labels, class_names = infer_classes_from_paths(image_paths)
    num_classes_original = len(class_names)
    
    print(f"   Classes detectadas: {num_classes_original}")
    print(f"   Total de imagens: {len(image_embeds)}")
    
    # ===== AGREGAÇÃO DE MÚLTIPLOS DESCRIPTORS =====
    class_descriptors = {}
    
    for class_name in class_names:
        class_code = class_name.split('-')[0] if '-' in class_name else class_name
        class_descriptors[class_name] = []
        
        if descriptions:
            for desc_key, desc_value in descriptions.items():
                if class_code in desc_key or class_name in desc_key:
                    class_descriptors[class_name].append(desc_value)
    
    # Estatísticas
    desc_counts = [len(descs) for descs in class_descriptors.values()]
    if desc_counts:
        print(f"   Descriptors: Total={sum(desc_counts)}, Média={np.mean(desc_counts):.1f}, Min={min(desc_counts)}, Max={max(desc_counts)}")
    
    # Gera lista flat de todos os textos
    all_texts = []
    text_to_class_idx = []
    
    for idx, class_name in enumerate(class_names):
        descs = class_descriptors.get(class_name, [])
        if not descs:
            descs = [f"a photo of a {class_name.replace('_', ' ')}"]
        
        all_texts.extend(descs)
        text_to_class_idx.extend([idx] * len(descs))
    
    # Gera embeddings de texto
    text_tokens = clip.tokenize(all_texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        all_text_embeds = model.encode_text(text_tokens)
        all_text_embeds /= all_text_embeds.norm(dim=-1, keepdim=True)
    
    # Agrupa por classe e faz MÉDIA
    print(f"🔄 Agregando {len(all_texts)} descriptors por classe...")
    final_text_embeds = []
    
    for idx in range(len(class_names)):
        indices = [i for i, c_idx in enumerate(text_to_class_idx) if c_idx == idx]
        class_embeds = all_text_embeds[indices]
        avg_embed = class_embeds.mean(dim=0)
        avg_embed /= avg_embed.norm()
        final_text_embeds.append(avg_embed)
    
    text_embeds = torch.stack(final_text_embeds)
    
    # Aplica filtering
    filtered_text_embeds, filtered_class_names, keep_indices = apply_comparative_filtering(
        text_embeds, class_names, threshold=SIMILARITY_THRESHOLD
    )
    
    # Filtra imagens e labels para manter apenas classes válidas
    valid_mask = np.isin(labels, keep_indices)
    filtered_image_embeds = image_embeds[valid_mask]
    filtered_labels = labels[valid_mask]
    
    # Remapeia labels
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    filtered_labels = np.array([old_to_new[label] for label in filtered_labels])
    
    print(f"📉 Imagens após filtragem: {len(filtered_labels)} (de {len(labels)})")
    
    # Avalia - GARANTE QUE TODOS ESTÃO NO MESMO DEVICE
    image_embeds_float = filtered_image_embeds.to(DEVICE).float()
    text_embeds_float = filtered_text_embeds.to(DEVICE).float()
    
    sims = image_embeds_float @ text_embeds_float.T
    preds = sims.argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(filtered_labels, preds)
    
    print(f"✅ Acurácia com filtering: {acc:.4f}")
    
    return acc, num_classes_original, len(filtered_class_names), len(filtered_labels)


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot com Comparative Filtering")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🔍 Threshold de similaridade: {SIMILARITY_THRESHOLD}\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "comparative filtering",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "total_datasets": len(DATASETS),
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            image_embeds, image_paths = load_embeddings(dataset_name)
            if image_embeds is None:
                print(f"⚠️  Pulando {dataset_name} (sem embeddings)")
                continue
            
            descriptions = load_descriptions(dataset_name)
            
            acc, num_orig, num_filtered, num_images = evaluate_with_filtering(
                dataset_name, image_embeds, image_paths, descriptions, model
            )
            
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes_original": num_orig,
                "num_classes_filtered": num_filtered,
                "num_images": num_images
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_filtering.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ Avaliação finalizada com sucesso.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()