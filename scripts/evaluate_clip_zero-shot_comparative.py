"""
Avaliação Zero-Shot COMPARATIVA com CLIP usando prompts simples.
Usa embeddings pré-calculados e templates básicos: "a photo of a {class}"
Serve como baseline para comparar com métodos usando descriptions.
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
RESULTS_DIR = "results_zero_shot_comparative"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_embeddings_and_generate_text(dataset_name, dataset_path, model):
    """Carrega embeddings de imagem e gera embeddings de texto com prompts simples"""
    
    embedding_path = os.path.join("embeddings", f"{dataset_name}.pt")
    
    if not os.path.exists(embedding_path):
        print(f"⚠️  Embeddings não encontrados: {embedding_path}")
        return None, None, None, None
    
    print(f"📂 Carregando embeddings: {embedding_path}")
    embeddings_data = torch.load(embedding_path, weights_only=False)
    
    if isinstance(embeddings_data, dict):
        image_embeds = embeddings_data['image_embeddings']
        image_paths = embeddings_data['image_paths']
    else:
        image_embeds = embeddings_data
        image_paths = None
    
    print(f"   Shape: {image_embeds.shape}")

    # Extrai classes e labels
    if image_paths:
        labels = []
        class_to_idx = {}
        class_names = []

        for path in image_paths:
            parts = Path(path).parts
            class_name = parts[-2] if len(parts) >= 2 else "unknown"

            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_names)
                class_names.append(class_name)

            labels.append(class_to_idx[class_name])
        
        labels = np.array(labels)
    else:
        print("❌ Nenhum path encontrado nos embeddings")
        return None, None, None, None

    print(f"   Total de imagens: {len(labels)} | Classes: {len(set(labels))}")

    # Gera text embeddings com PROMPTS SIMPLES (baseline)
    class_texts = [f"a photo of a {class_name.replace('_', ' ')}" for class_name in class_names]

    print(f"\n📝 Gerando text embeddings para {len(class_texts)} classes...")
    print(f"   Usando prompts simples (baseline)")
    print(f"   Exemplo: {class_texts[0]}")

    # Tokeniza e gera embeddings com CLIP
    text_tokens = clip.tokenize(class_texts, truncate=True).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    print(f"✅ Text embeddings prontos! Shape: {text_embeds.shape}")
    
    return image_embeds, text_embeds.cpu(), labels, class_names


def evaluate_zero_shot(image_embeds, text_embeds, labels):
    """Calcula acurácia zero-shot."""
    # Garante que ambos estão em float32
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    
    sims = image_embeds @ text_embeds.T
    preds = sims.argmax(dim=-1).numpy()
    acc = accuracy_score(labels, preds)
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='viridis', aspect='auto')
    plt.title("CLIP Zero-Shot (Baseline - Simple Prompts)", fontsize=14)
    plt.colorbar()
    fontsize = max(6, 12 - len(class_names) // 10)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=fontsize)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot COMPARATIVA (Baseline)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"📝 Método: Prompts simples - 'a photo of a {{class}}'")
    print(f"🎯 Objetivo: Baseline para comparação com descriptions\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "CLIP baseline (simple prompts)",
        "prompt_template": "a photo of a {class}",
        "total_datasets": len(DATASETS),
        "successful": 0,
        "failed": 0,
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📊 Avaliando dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            result = load_embeddings_and_generate_text(
                dataset_name, dataset_path, model
            )
            
            if result[0] is None:
                print(f"⏭️  Pulando {dataset_name}")
                summary["failed"] += 1
                continue
                
            image_embeds, text_embeds, labels, class_names = result

            acc, preds = evaluate_zero_shot(image_embeds, text_embeds, labels)
            print(f"\n✅ Acurácia zero-shot (BASELINE): {acc:.4f}")

            plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_cm.png")
            plot_confusion_matrix(labels, preds, class_names, plot_path)

            summary["successful"] += 1
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes": len(class_names),
                "num_images": len(labels),
                "confusion_matrix_plot": plot_path
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            summary["failed"] += 1
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_comparative.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ {summary['successful']} datasets processados com sucesso.")
    print(f"❌ {summary['failed']} falharam.")
    print(f"\n💡 Este é o BASELINE para comparação com descriptions!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()