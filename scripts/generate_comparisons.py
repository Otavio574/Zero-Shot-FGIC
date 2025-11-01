"""
Gera descritores comparativos entre classes (comparative descriptors)
para uso com ComparativeCLIP.
"""

import os
import json
import random
from pathlib import Path

# ==========================
# CONFIGURAÇÕES
# ==========================
DATASET_NAME = "Stanford_Dogs"
DATASET_PATH = Path("datasets") / DATASET_NAME
OUTPUT_PATH = Path("descriptors") / f"{DATASET_NAME}_descriptors_comparative.json"

NUM_COMPARISONS_PER_CLASS = 3  # número de frases comparativas por classe
SEED = 42
random.seed(SEED)

# ==========================
# FUNÇÕES
# ==========================
def extract_class_names(dataset_path: Path):
    """Extrai nomes das pastas (classes)"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {dataset_path}")
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    print(f"✅ {len(classes)} classes detectadas.")
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


def main():
    print(f"🐶 Gerando descritores comparativos para {DATASET_NAME}...")
    os.makedirs("descriptors", exist_ok=True)

    classes = extract_class_names(DATASET_PATH)
    descriptors = generate_comparative_descriptors(classes, NUM_COMPARISONS_PER_CLASS)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(descriptors, f, indent=4, ensure_ascii=False)

    print(f"✅ Descritores comparativos salvos em: {OUTPUT_PATH}")
    print(f"📊 Total de classes: {len(descriptors)}")
    print(f"💬 Exemplos de comparações para '{classes[0]}':")
    for ex in descriptors[classes[0]][:3]:
        print(f"   - {ex}")


if __name__ == "__main__":
    main()
