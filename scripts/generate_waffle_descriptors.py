import json, random, os
from pathlib import Path

# Caminho para seu arquivo de descrições base
SOURCE_PATH = Path("descriptors/Stanford_Dogs_descriptors.json")
OUTPUT_PATH = Path("descriptors/Stanford_Dogs_waffle.json")

# Lista de palavras aleatórias (pode aumentar se quiser)
RANDOM_WORDS = [
    "Humvee", "banana", "cucumber", "airplane", "candle", "motorcycle",
    "keyboard", "pizza", "toaster", "telescope", "lighthouse",
    "cactus", "guitar", "shark", "pencil", "umbrella", "volcano"
]

N_RANDOM_WORDS = 3  # Quantas palavras ruído incluir por frase

# Carrega descrições originais
with open(SOURCE_PATH, "r", encoding="utf-8") as f:
    base_desc = json.load(f)

waffle_desc = {}

for cls, text in base_desc.items():
    waffle_desc[cls] = []
    for _ in range(3):  # gera 3 variações por classe
        noise = ", ".join(random.sample(RANDOM_WORDS, N_RANDOM_WORDS))
        phrase = f"{text}, {noise}"
        waffle_desc[cls].append(phrase)

# Salva o novo JSON
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(waffle_desc, f, indent=4, ensure_ascii=False)

print(f"✅ Waffle descriptors gerados e salvos em {OUTPUT_PATH}")
