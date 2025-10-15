# Zero-Shot FGIC - CLIP (HuggingFace)

## Objetivo
Pipeline para avaliar CLIP (zero-shot) em datasets fine-grained. Inicialmente configurado para Stanford Dogs.

## Requisitos
1. Tenha CUDA + drivers instalados para usar GPU.
2. Crie ambiente virtual:

python -m venv venv
source venv/bin/activate # linux/mac
venv\Scripts\activate # windows
pip install -r requirements.txt


## Organização do dataset
Coloque o dataset (Stanford Dogs) em:

/path/to/stanford_dogs/
/Chihuahua/
img001.jpg
/Labrador_retriever/
img010.jpg

As pastas devem ser os nomes das classes (spaces ou underscores são aceitos).

## Como rodar

python -m src.main --dataset /path/to/stanford_dogs --out dogs_predictions.csv --batch 32


## Observações
- Para economizar memória, use `--batch 16` se der OOM.
- O script gera `dogs_predictions.csv` com colunas: true_class, top1, top1_score, top5, top1_hit, top5_hit.
- Para experimentar templates diferentes, edite `DEFAULT_TEMPLATES` em `src/generate_prompts.py`.
