"""
Gera descritores comparativos (comparative descriptors) para todos os datasets
na pasta datasets/. Cada dataset deve conter subpastas representando classes.

Saída: descriptors/{dataset_name}_descriptors_comparative.json
"""

import os
import json
import random
from pathlib import Path

# ==========================
# CONFIGURAÇÕES
# ==========================
DATASETS_DIR = Path("datasets")
OUTPUT_DIR = Path("descriptors")
NUM_COMPARISONS_PER_CLASS = 3
SEED = 42
random.seed(SEED)

# ==========================
# FUNÇÕES
# ==========================

def extract_class_names(dataset_path: Path):
    """Extrai os nomes das classes (subpastas)"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {dataset_path}")
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    return classes


def generate_comparison(breed_a, breed_b):
    """Gera uma frase comparando duas classes"""
    templates = [
        f"A {breed_a} is smaller than a {breed_b}.",
        f"A {breed_a} has a different coat pattern compared to a {breed_b}.",
        f"A {breed_a} usually has a different body shape than a {breed_b}.",
        f"A {breed_a} looks more delicate than a {breed_b}.",
        f"A {breed_a} has larger ears than a {breed_b}.",
        f"A {breed_a} is more muscular than a {breed_b}.",
        f"A {breed_a} has a shorter muzzle than a {breed_b}.",
        f"A {breed_a} tends to have longer fur than a {breed_b}.",
        f"A {breed_a} appears more slender than a {breed_b}.",
        f"A {breed_a} is heavier and sturdier than a {breed_b}.",
    ]
    return random.choice(templates)


def generate_comparative_descriptors(classes, n_per_class=3):
    """Gera descritores comparativos para todas as classes"""
    descriptors = {}
    for breed_a in classes:
        others = [b for b in classes if b != breed_a]
        random.shuffle(others)
        selected = others[:n_per_class]
        phrases = [generate_comparison(breed_a, b) for b in selected]
        descriptors[breed_a] = phrases
    return descriptors


# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================

def main():
    print("🐾 Gerando descritores comparativos para todos os datasets...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = [
        d for d in DATASETS_DIR.iterdir()
        if d.is_dir() and d.name in ["Stanford_Dogs", "CUB_200_2011", "Food101"]
    ]

    if not datasets:
        print("❌ Nenhum dataset encontrado em 'datasets/'.")
        return

    for dataset_path in datasets:
        dataset_name = dataset_path.name
        print(f"📊 Processando {dataset_name}...")

        try:
            classes = extract_class_names(dataset_path)
            print(f"   ✅ {len(classes)} classes detectadas.")

            descriptors = generate_comparative_descriptors(classes, NUM_COMPARISONS_PER_CLASS)

            output_file = OUTPUT_DIR / f"{dataset_name}_descriptors_comparative.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(descriptors, f, indent=4, ensure_ascii=False)

            print(f"   💾 Arquivo salvo em: {output_file}")
            print(f"   💬 Exemplo para '{classes[0]}': {descriptors[classes[0]][0]}")
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")

    print("\n🎯 Geração de descritores comparativos concluída com sucesso!")


if __name__ == "__main__":
    main()
