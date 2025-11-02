"""
Avaliação Zero-Shot de CLIP usando embeddings pré-calculados e descriptors.
Este script carrega embeddings de imagens já extraídos, gera embeddings de texto
a partir dos descriptors CLIP, e avalia a acurácia zero-shot.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
from pathlib import Path
from glob import glob
from collections import Counter

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

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_baseline"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptors(dataset_name):
    """Carrega descriptors do dataset"""
    path = os.path.join("descriptors", f"{dataset_name}_descriptors.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_and_generate_text(dataset_name, dataset_path, descriptors, model, processor):
    """Carrega embeddings de imagem e gera embeddings de texto dos descriptors"""
    
    # Carrega image embeddings
    embedding_path = os.path.join("embeddings", f"{dataset_name}.pt")
    
    if not os.path.exists(embedding_path):
        print(f"⚠️  Embeddings não encontrados: {embedding_path}")
        return None, None, None, None
    
    print(f"📂 Carregando embeddings: {embedding_path}")
    embeddings_data = torch.load(embedding_path)
    
    # Detecta formato
    if isinstance(embeddings_data, dict):
        # Formato: {'image_embeddings': tensor, 'image_paths': list}
        image_embeds = embeddings_data['image_embeddings']
        image_paths = embeddings_data['image_paths']
        print(f"   Formato: dicionário com paths")
    else:
        # Formato: apenas tensor
        image_embeds = embeddings_data
        image_paths = None
        print(f"   Formato: tensor direto")
    
    print(f"   Shape: {image_embeds.shape}")
    
    # Extrai classes e labels
    if image_paths:
        # Usa paths para extrair classes
        labels = []
        class_to_idx = {}
        class_names = []
        
        for path in image_paths:
            # Extrai nome da classe do path (assume formato: .../class_name/image.jpg)
            parts = Path(path).parts
            # Procura pela pasta que contém a imagem (penúltima parte)
            class_name = parts[-2] if len(parts) >= 2 else "unknown"
            
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_names)
                class_names.append(class_name)
            
            labels.append(class_to_idx[class_name])
        
        labels = np.array(labels)
    else:
        # Sem paths salvos - precisa descobrir pela estrutura
        print("⚠️  Sem paths salvos, tentando inferir da estrutura de pastas...")
        
        class_folders = {}
        
        # Busca recursiva PROFUNDA (até 5 níveis para pegar Birdsnap e CompCars)
        for depth in range(1, 6):  # 1 a 5 níveis
            pattern = os.path.join(dataset_path, *['*'] * depth)
            
            for potential_dir in glob(pattern):
                if not os.path.isdir(potential_dir):
                    continue
                
                # Verifica se tem imagens diretamente
                imgs = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    imgs.extend(glob(os.path.join(potential_dir, ext)))
                
                if imgs:
                    # Usa o nome da pasta mais específica (última parte do path)
                    class_name = os.path.basename(potential_dir)
                    
                    # Evita duplicatas
                    if class_name not in class_folders:
                        class_folders[class_name] = []
                    class_folders[class_name].extend(imgs)
        
        if not class_folders:
            print("❌ Não foi possível inferir classes")
            return None, None, None, None
        
        class_names = sorted(class_folders.keys())
        print(f"   Classes encontradas: {len(class_names)}")
        print(f"   Primeiras 5: {class_names[:5]}")
        
        # Cria labels baseado na distribuição uniforme
        labels = []
        imgs_per_class = len(image_embeds) // len(class_names)
        
        for i in range(len(image_embeds)):
            class_idx = min(i // imgs_per_class, len(class_names) - 1)
            labels.append(class_idx)
            
        labels = np.array(labels)
        print(f"   ⚠️  Labels inferidos por distribuição uniforme (pode não ser perfeito)")
    
    print(f"   Total de imagens: {len(labels)}")
    print(f"   Distribuição de classes:")
    class_counts = Counter(labels)
    for cls_idx, count in list(class_counts.items())[:5]:
        print(f"      {class_names[cls_idx]}: {count} imagens")
    
    # Gera text embeddings dos descriptors
    class_texts = []
    for class_name in class_names:
        # Busca primeira descrição dessa classe nos descriptors
        found = False
        for filename, description in descriptors.items():
            # Verifica se o filename pertence a essa classe (busca flexível)
            if class_name.lower().replace('_', ' ') in filename.lower() or \
               class_name.lower().replace('-', ' ') in filename.lower():
                class_texts.append(description)
                found = True
                break
        
        if not found:
            # Fallback: template genérico
            class_texts.append(f"a photo of a {class_name.replace('_', ' ').replace('-', ' ')}")
    
    print(f"\n📝 Gerando text embeddings para {len(class_texts)} classes...")
    print(f"   Exemplos de textos:")
    for i, txt in enumerate(class_texts[:3]):
        print(f"      {class_names[i]}: {txt[:70]}...")
    
    text_inputs = processor(
        text=class_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    
    print(f"✅ Embeddings carregados e processados!")
    
    return image_embeds, text_embeds.cpu(), labels, class_names


def evaluate_zero_shot(image_embeds, text_embeds, labels):
    """Calcula acurácia zero-shot."""
    sims = image_embeds @ text_embeds.T
    preds = sims.argmax(dim=-1).numpy()
    acc = accuracy_score(labels, preds)

    print(f"DEBUG - Text embeds shape: {text_embeds.shape}")
    print(f"DEBUG - Text embeds sample: {text_embeds[0][:5]}")  # Primeiros 5 valores
    print(f"DEBUG - Labels unique: {np.unique(labels)}")
    
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    cm = confusion_matrix(labels, preds, normalize='true')
    
    # Limita o número de classes para visualização (se for muito grande)
    max_classes = 50
    if len(class_names) > max_classes:
        print(f"   ⚠️  Muitas classes ({len(class_names)}), mostrando top {max_classes}")
        # Pega as classes mais frequentes
        unique, counts = np.unique(labels, return_counts=True)
        top_classes = unique[np.argsort(counts)[-max_classes:]]
        
        # Filtra para mostrar apenas essas classes
        mask = np.isin(labels, top_classes)
        labels_filtered = labels[mask]
        preds_filtered = preds[mask]
        class_names_filtered = [class_names[i] for i in top_classes]
        
        cm = confusion_matrix(labels_filtered, preds_filtered, normalize='true')
        class_names = class_names_filtered
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='viridis', aspect='auto')
    plt.title("Zero-Shot Confusion Matrix", fontsize=14)
    plt.colorbar()
    
    # Ajusta tamanho da fonte baseado no número de classes
    fontsize = max(6, 12 - len(class_names) // 10)
    
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=fontsize)
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Iniciando avaliação Zero-Shot CLIP")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"📊 Datasets: {len(DATASETS)}\n")
    
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    summary = {
        "model": MODEL_NAME,
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
            descriptors = load_descriptors(dataset_name)
            
            # USA EMBEDDINGS PRÉ-CALCULADOS!
            result = load_embeddings_and_generate_text(
                dataset_name, dataset_path, descriptors, model, processor
            )
            
            if result[0] is None:
                print(f"⏭️  Pulando {dataset_name}")
                summary["failed"] += 1
                continue
                
            image_embeds, text_embeds, labels, class_names = result

            acc, preds = evaluate_zero_shot(image_embeds, text_embeds, labels)
            print(f"\n✅ Acurácia zero-shot: {acc:.4f}")

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

    # Salva resultados detalhados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Salva sumário simples (apenas acurácias)
    summary_path = os.path.join(RESULTS_DIR, "accuracy_summary.json")
    accuracy_only = {name: f"{data['accuracy']:.4f}" 
                     for name, data in summary["results"].items()}
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_only, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMO FINAL")
    print(f"{'='*60}")
    print(f"✅ Datasets processados com sucesso: {summary['successful']}")
    print(f"❌ Datasets com falha: {summary['failed']}")
    print(f"\n📈 Acurácias (ordenadas):")
    
    for name, data in sorted(summary["results"].items(), 
                             key=lambda x: x[1]["accuracy"], 
                             reverse=True):
        print(f"   {name:30s}: {data['accuracy']:.4f} ({data['num_classes']} classes, {data['num_images']} imgs)")
    
    print(f"\n📁 Resultados salvos em:")
    print(f"   - {out_path}")
    print(f"   - {summary_path}")
    print(f"   - Matrizes de confusão em: {RESULTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()