
'''
import os
import pandas as pd

# Caminho onde você descompactou o dataset
DATASET_DIR = r"C:\Users\Otavio Augusto\stanford_dogs\images\Images"

# Listas para armazenar dados
image_paths = []
labels = []

# Percorrer cada pasta de classe
for class_folder in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue
    
    # Remover prefixo do código (n02085620-)
    class_name = class_folder.split('-', 1)[-1] if '-' in class_folder else class_folder

    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)
            labels.append(class_name)

# Criar DataFrame
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Salvar CSV (opcional)
df.to_csv("stanford_dogs_labels.csv", index=False)

print(f"[INFO] Dataset preparado! Total de imagens: {len(df)}")
print(df.head())
'''