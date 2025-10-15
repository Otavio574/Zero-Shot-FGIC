# src/load_data.py
from pathlib import Path
from typing import List, Tuple

def list_image_paths(dataset_root: str) -> List[Tuple[str, str]]:
    """
    Retorna lista de (image_path, class_name).
    Espera estrutura: dataset_root/class_name/*.jpg
    """
    root = Path(dataset_root)
    items = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        class_name = class_dir.name
        for img in sorted(class_dir.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((str(img), class_name))
    return items

def build_class_names(dataset_root: str) -> List[str]:
    """Lista de classes (nomes das pastas)"""
    root = Path(dataset_root)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])
