"""
Avaliação Zero-Shot com CLIP usando descriptions detalhadas.
Usa embeddings pré-calculados e descriptions das classes.
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

path_string = "all_zero-shot_results/"

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "ViT-B/32"  # Modelo CLIP do OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = path_string + "results_zero_shot_description"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """Carrega descriptions do dataset"""
    path = os.path.join("descriptors", f"{dataset_name}_descriptors.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_and_generate_text(dataset_name, dataset_path, descriptions, model):
    """Carrega embeddings de imagem e gera embeddings de texto das descriptions"""
    
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
        # Busca profunda por classes
        print("⚠️  Sem paths salvos, tentando inferir classes...")
        class_folders = {}

        for depth in range(1, 6):
            pattern = os.path.join(dataset_path, *['*'] * depth)
            for potential_dir in glob(pattern):
                if not os.path.isdir(potential_dir):
                    continue
                
                imgs = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    imgs.extend(glob(os.path.join(potential_dir, ext)))
                
                if imgs:
                    class_name = os.path.basename(potential_dir)
                    if class_name not in class_folders:
                        class_folders[class_name] = []
                    class_folders[class_name].extend(imgs)

        if not class_folders:
            print("❌ Não foi possível inferir classes")
            return None, None, None, None

        class_names = sorted(class_folders.keys())
        imgs_per_class = len(image_embeds) // len(class_names)
        labels = np.array([min(i // imgs_per_class, len(class_names) - 1) for i in range(len(image_embeds))])
        print(f"   ⚠️  Labels inferidos automaticamente.")

    print(f"   Total de imagens: {len(labels)} | Classes: {len(set(labels))}")

    # Gera text embeddings das descriptions
    class_texts = []
    for class_name in class_names:
        found = False
        for filename, description in descriptions.items():
            # Compara nome da classe com o arquivo da description
            if class_name.lower().replace('_', ' ') in filename.lower():
                class_texts.append(description)
                found = True
                break
        if not found:
            # Fallback: descrição simples se não encontrar description
            class_texts.append(f"a photo of a {class_name.replace('_', ' ')}")

    print(f"\n📝 Gerando text embeddings para {len(class_texts)} classes...")
    print(f"   Exemplo de description: {class_texts[0][:100]}...")

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
    
    # ====== DEBUG ======
    print(f"\n🔍 DEBUG:")
    print(f"   Image embeds shape: {image_embeds.shape}")
    print(f"   Text embeds shape: {text_embeds.shape}")
    print(f"   Text embeds sample [0][:5]: {text_embeds[0][:5]}")
    print(f"   Labels unique: {np.unique(labels)}")
    # ==================
    
    sims = image_embeds @ text_embeds.T
    preds = sims.argmax(dim=-1).numpy()
    acc = accuracy_score(labels, preds)
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='viridis', aspect='auto')
    plt.title("CLIP Zero-Shot with Descriptions", fontsize=14)
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
    print(f"🚀 Avaliação Zero-Shot com CLIP + Descriptions")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "CLIP with descriptions",
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
            descriptions = load_descriptions(dataset_name)
            print(f"✅ Descriptions carregadas: {len(descriptions)} classes")
            
            result = load_embeddings_and_generate_text(
                dataset_name, dataset_path, descriptions, model
            )
            
            if result[0] is None:
                print(f"⏭️  Pulando {dataset_name}")
                summary["failed"] += 1
                continue
                
            image_embeds, text_embeds, labels, class_names = result

            acc, preds = evaluate_zero_shot(image_embeds, text_embeds, labels)
            print(f"\n✅ Acurácia zero-shot (CLIP + Descriptions): {acc:.4f}")

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
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_description.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ {summary['successful']} datasets processados com sucesso.")
    print(f"❌ {summary['failed']} falharam.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()