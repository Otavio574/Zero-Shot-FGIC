import os
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# ============================
# CONFIGURAÇÕES GERAIS
# ============================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS_DIR = Path("datasets")
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================
# CARREGAR MODELO CLIP
# ============================

print("🚀 Carregando modelo CLIP (baseline)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ============================
# CARREGAR LISTA DE DATASETS
# ============================

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    datasets = json.load(f)

# ============================
# PRÉ-PROCESSAMENTO PADRÃO
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================
# FUNÇÕES AUXILIARES
# ============================

def get_image_paths(folder, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    """Busca recursiva de imagens em subpastas"""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


def generate_embeddings_for_dataset(dataset_name: str, dataset_path: Path, limit: int = 5000):
    """Gera e salva embeddings de imagem para um dataset"""
    print(f"\n📦 Processando dataset: {dataset_name}")
    image_paths = get_image_paths(dataset_path)

    if len(image_paths) == 0:
        print(f"⚠️ Nenhuma imagem encontrada em {dataset_path}")
        return False

    all_embeds = []
    valid_paths = []

    for img_path in tqdm(image_paths[:limit], desc=f"🔹 {dataset_name}"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore
            with torch.no_grad():
                embeds = model.get_image_features(**inputs)
                embeds /= embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())
            valid_paths.append(img_path)
        except Exception as e:
            continue

    if all_embeds:
        all_embeds = torch.cat(all_embeds)

        out_path = OUTPUT_DIR / f"{dataset_name}.pt"
        torch.save({
            "image_embeddings": all_embeds,
            "image_paths": valid_paths
        }, out_path)

        print(f"💾 Embeddings salvos em {out_path} ({len(all_embeds)} imagens).")
        return True
    else:
        print(f"⚠️ Nenhuma embedding gerada para {dataset_name}")
        return False

# ============================
# LOOP PRINCIPAL INCREMENTAL
# ============================

def main():
    print("\n🔍 Verificando datasets...")
    existing_files = {p.stem for p in OUTPUT_DIR.glob("*.pt")}

    total = len(datasets)
    new_datasets = [ds for ds in datasets if ds["dataset"] not in existing_files]

    print(f"📊 Total de datasets no summary: {total}")
    print(f"✅ Já processados: {len(existing_files)}")
    print(f"🚀 Restantes: {len(new_datasets)} → {[d['dataset'] for d in new_datasets]}")

    for ds in new_datasets:
        name = ds["dataset"]
        path = Path(ds["path"])

        if not path.exists():
            print(f"⚠️ Caminho inválido: {path}")
            continue

        success = generate_embeddings_for_dataset(name, path)
        if not success:
            print(f"❌ Falha ao gerar embeddings para {name}")

    print("\n🏁 Finalizado! Embeddings atualizados em /embeddings.")


if __name__ == "__main__":
    main()
