"""
Gera descrições de imagens usando CLIP e salva em formato simples:
{
  "image_001.jpg": "description of the image",
  "image_002.jpg": "another description"
}
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
import torch
import clip

class CLIPDescriptorGenerator:
    """Gera descrições usando CLIP via template matching"""
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.descriptions = None
        
    def load_model(self):
        """Carrega o modelo CLIP"""
        print(f"⏳ Carregando CLIP: {self.model_name}...")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        print("✅ CLIP carregado!")
    
    def prepare_descriptions(self, concepts: List[str], templates: Optional[List[str]] = None):
        """
        Prepara o conjunto de descrições possíveis.
        
        Args:
            concepts: Lista de conceitos (ex: ["dog", "cat", "car"])
            templates: Templates opcionais (usa padrão se None)
        """
        if templates is None:
            templates = [
                "a photo of a {}",
                "a close-up photo of a {}",
                "a picture of a {}",
                "an image showing a {}",
                "a photograph of a {}",
                "{} in the image",
                "a {} in the scene",
                "a view of a {}",
            ]
        
        print(f"📝 Preparando descrições...")
        print(f"   Conceitos: {len(concepts)}")
        print(f"   Templates: {len(templates)}")
        
        # Gera todas as combinações
        self.descriptions = []
        for concept in concepts:
            for template in templates:
                self.descriptions.append(template.format(concept))
        
        print(f"   Total: {len(self.descriptions)} descrições possíveis")
        
        # Pré-computa embeddings de texto
        print("⏳ Encodando descrições com CLIP...")
        text_tokens = clip.tokenize(self.descriptions).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        print("✅ Descrições prontas!")
    
    def describe_image(self, image_path: str) -> str:
        """Retorna a melhor descrição para uma imagem"""
        if self.text_features is None:
            raise ValueError("Use prepare_descriptions() antes de descrever imagens!")
        
        # Processa imagem
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Computa similaridade
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ self.text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()
        
        return self.descriptions[best_idx]
    
    def process_dataset(self, 
                       image_paths: List[str],
                       output_path: str,
                       max_images: Optional[int] = None) -> Dict[str, str]:
        """
        Processa dataset e retorna dicionário {filename: description}
        
        Args:
            image_paths: Lista de caminhos completos das imagens
            output_path: Onde salvar o JSON
            max_images: Limitar número de imagens (None = todas)
            
        Returns:
            Dicionário com {filename: description}
        """
        if max_images:
            image_paths = image_paths[:max_images]
        
        descriptors = {}
        failed = []
        
        print(f"\n🎨 Gerando descrições para {len(image_paths)} imagens...")
        
        for img_path in tqdm(image_paths, desc="Processando"):
            try:
                # Pega apenas o nome do arquivo (sem path)
                filename = os.path.basename(img_path)
                
                # Gera descrição
                description = self.describe_image(img_path)
                
                descriptors[filename] = description
                
            except Exception as e:
                failed.append({"path": img_path, "error": str(e)})
                print(f"\n❌ Erro em {img_path}: {e}")
        
        # Salva JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Concluído!")
        print(f"   Sucessos: {len(descriptors)}")
        print(f"   Falhas: {len(failed)}")
        print(f"   Salvo em: {output_path}")
        
        # Salva erros se houver
        if failed:
            error_path = output_path.replace('.json', '_errors.json')
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(failed, f, indent=2, ensure_ascii=False)
            print(f"   Erros em: {error_path}")
        
        return descriptors


def load_image_paths_from_folder(folder_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> List[str]:
    """Carrega todos os paths de imagens de uma pasta"""
    folder = Path(folder_path)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(folder.rglob(f'*{ext}'))
        image_paths.extend(folder.rglob(f'*{ext.upper()}'))
    
    return [str(p) for p in sorted(image_paths)]


def get_concepts_for_dataset(dataset_name: str) -> List[str]:
    """Retorna conceitos apropriados para cada dataset"""
    
    concepts_map = {
        # Birds
        "CUB-200-2011": [
            "bird", "avian", "songbird", "waterfowl", "raptor", "seabird",
            "warbler", "sparrow", "finch", "thrush", "wren", "robin",
            "cardinal", "jay", "crow", "raven", "blackbird", "oriole",
            "hummingbird", "woodpecker", "flycatcher", "swallow", "swift",
            "hawk", "eagle", "falcon", "owl", "vulture", "kite",
            "duck", "goose", "swan", "heron", "egret", "pelican",
            "gull", "tern", "sandpiper", "plover", "grebe",
        ],
        "Birdsnap": [
            "bird", "avian", "songbird", "perching bird", "flying bird",
            "sparrow", "finch", "warbler", "thrush", "wren", "chickadee",
            "titmouse", "nuthatch", "creeper", "cardinal", "grosbeak",
            "bunting", "tanager", "oriole", "blackbird", "grackle",
        ],
        
        # Dogs
        "Stanford Dogs": [
            "dog", "canine", "puppy", "pet", "domestic dog", "breed",
            "retriever", "labrador", "golden retriever", "german shepherd",
            "bulldog", "poodle", "beagle", "chihuahua", "husky", "corgi",
            "terrier", "spaniel", "setter", "pointer", "hound", "mastiff",
            "schnauzer", "boxer", "dalmatian", "dachshund", "pug",
        ],
        
        # Pets (dogs + cats)
        "Oxford-IIIT Pets": [
            "pet", "domestic animal", "companion animal",
            "dog", "canine", "puppy", "breed dog",
            "cat", "feline", "kitten", "tabby cat", "persian cat",
            "british shorthair", "siamese cat", "maine coon",
            "pomeranian", "yorkshire terrier", "shiba inu", "samoyed",
        ],
        
        # Cars
        "Stanford Cars": [
            "car", "automobile", "vehicle", "sedan", "coupe", "suv",
            "convertible", "hatchback", "wagon", "sports car", "luxury car",
            "compact car", "minivan", "pickup truck", "crossover",
            "bmw", "mercedes", "audi", "toyota", "honda", "ford",
        ],
        "CompCars": [
            "car", "vehicle", "automobile", "motor vehicle", "sedan",
            "suv", "hatchback", "coupe", "convertible", "sports car",
            "luxury vehicle", "compact car", "family car", "truck",
        ],
        
        # Aircraft
        "FGVC Aircraft": [
            "aircraft", "airplane", "plane", "jet", "airliner", "fighter",
            "boeing", "airbus", "cessna", "commercial aircraft", "military aircraft",
            "passenger plane", "cargo plane", "propeller plane", "biplane",
            "helicopter", "turboprop", "jumbo jet", "fighter jet",
        ],
        
        # Flowers
        "Oxford Flowers 102": [
            "flower", "bloom", "blossom", "petal", "floral", "plant",
            "rose", "tulip", "daisy", "sunflower", "lily", "orchid",
            "carnation", "iris", "daffodil", "poppy", "peony", "dahlia",
            "hibiscus", "marigold", "chrysanthemum", "lavender",
        ],
        
        # Leaves
        "Flavia Leaves": [
            "leaf", "foliage", "plant leaf", "tree leaf", "botanical",
            "maple leaf", "oak leaf", "birch leaf", "elm leaf",
            "green leaf", "autumn leaf", "veined leaf", "serrated leaf",
        ],
        
        # Food
        "Food-101": [
            "food", "meal", "dish", "cuisine", "plate", "dessert", "snack",
            "pizza", "burger", "hamburger", "sushi", "pasta", "salad",
            "steak", "chicken", "fish", "soup", "sandwich", "taco",
            "cake", "ice cream", "pie", "cookie", "donut", "bread",
            "rice", "noodles", "curry", "fries", "pancake", "waffle",
        ],
        
        # Nature/Wildlife
        "iNaturalist19": [
            "animal", "plant", "organism", "wildlife", "nature", "species",
            "mammal", "bird", "reptile", "amphibian", "fish", "insect",
            "arachnid", "mollusk", "fungi", "mushroom", "lichen",
            "tree", "flower", "grass", "fern", "moss", "algae",
        ],
        
        # Insects - Butterflies
        "Butterfly MNIST": [
            "butterfly", "moth", "insect", "lepidoptera", "winged insect",
            "monarch butterfly", "swallowtail", "admiral", "fritillary",
            "painted lady", "blue butterfly", "skipper", "hairstreak",
        ],
        
        # Insects - Bees
        "Bee Images Dataset": [
            "bee", "honeybee", "bumblebee", "insect", "pollinator",
            "worker bee", "queen bee", "drone bee", "flying insect",
            "buzzing insect", "apis mellifera", "hymenoptera",
        ],
        
        # Fish
        "Fish Recognition (Kaggle)": [
            "fish", "aquatic animal", "marine life", "seafood", "underwater animal",
            "salmon", "tuna", "trout", "bass", "carp", "catfish",
            "goldfish", "shark", "ray", "eel", "mackerel", "herring",
            "cod", "snapper", "grouper", "perch", "pike", "tilapia",
        ],
    }
    
    # Tenta encontrar o dataset (case-insensitive e com flexibilidade)
    dataset_normalized = dataset_name.lower().replace("-", " ").replace("_", " ")
    
    for key, concepts in concepts_map.items():
        key_normalized = key.lower().replace("-", " ").replace("_", " ")
        if key_normalized in dataset_normalized or dataset_normalized in key_normalized:
            return concepts
    
    # Fallback: conceitos genéricos
    print(f"⚠️  Dataset '{dataset_name}' não encontrado no mapa, usando conceitos genéricos")
    return [
        "object", "animal", "plant", "vehicle", "food", "tool",
        "bird", "dog", "cat", "fish", "insect", "flower", "tree",
        "car", "airplane", "building", "furniture", "person",
    ]


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


def generate_descriptors_for_datasets(
    datasets_config: Dict[str, str],
    output_dir: str = "descriptors",
    max_images: Optional[int] = None,
    clip_model: str = "ViT-B/32"
):
    """
    Gera descriptors para múltiplos datasets.
    
    Args:
        datasets_config: Dict com {nome_dataset: pasta_das_imagens}
        output_dir: Pasta onde salvar os JSONs
        max_images: Limite por dataset (None = todas)
        clip_model: Modelo CLIP a usar
    """
    
    # Inicializa gerador
    generator = CLIPDescriptorGenerator(model_name=clip_model)
    generator.load_model()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando geração de descriptors")
    print(f"📊 Datasets: {len(datasets_config)}")
    print(f"🔧 Modelo: {clip_model}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for dataset_name, dataset_path in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"📁 Dataset: {dataset_name}")
        
        # Carrega imagens
        image_paths = load_image_paths_from_folder(dataset_path)
        print(f"🖼️  Imagens encontradas: {len(image_paths)}")
        
        if not image_paths:
            print(f"⚠️  Nenhuma imagem encontrada em {dataset_path}")
            continue
        
        # Prepara descrições específicas para o dataset
        concepts = get_concepts_for_dataset(dataset_name)
        generator.prepare_descriptions(concepts)
        
        # Processa
        output_path = os.path.join(output_dir, f"{dataset_name}_descriptors.json")
        generator.process_dataset(
            image_paths=image_paths,
            output_path=output_path,
            max_images=max_images
        )
    
    print(f"\n{'='*60}")
    print(f"✅ Todos os descriptors gerados!")
    print(f"📂 Verifique a pasta: {output_dir}")
    print(f"{'='*60}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    from pathlib import Path
    
    # Configuração dos paths
    SUMMARY_PATH = Path("outputs/analysis/summary.json")
    OUTPUT_DIR = "descriptors"
    
    # Carrega datasets automaticamente do summary.json
    print("📂 Carregando configuração dos datasets...")
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset encontrado no summary!")
        print(f"   Verifique se o arquivo existe: {SUMMARY_PATH}")
        exit(1)
    
    # Gera descriptors para todos os datasets
    generate_descriptors_for_datasets(
        datasets_config=datasets,
        output_dir=OUTPUT_DIR,
        max_images=None,  # None para processar todas as imagens, ou um número para limitar
        clip_model="ViT-B/32"  # ou "ViT-L/14" para melhor qualidade (mais lento)
    )
    
    print("\n🎉 Descriptors prontos para uso no zero-shot!")
    print(f"📂 Verifique a pasta: {OUTPUT_DIR}/")
    
    # Exemplo de como carregar e usar depois:
    # with open("descriptors/Bee_Images_Dataset_descriptors.json", "r", encoding="utf-8") as f:
    #     descriptors = json.load(f)
    #     for filename, description in list(descriptors.items())[:3]:
    #         print(f"{filename}: {description}")