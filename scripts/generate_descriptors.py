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
    
    def process_dataset_by_class(self, 
                                 dataset_path: str,
                                 output_path: str,
                                 max_images_per_class: Optional[int] = None) -> Dict[str, str]:
        """
        Processa dataset ORGANIZADO POR CLASSES e gera UM descriptor por classe.
        
        Estrutura esperada:
        dataset_path/
        ├── n02085620-Chihuahua/
        │   ├── img1.jpg
        │   ├── img2.jpg
        └── n02085782-Japanese_spaniel/
            ├── img1.jpg
            └── img2.jpg
        
        Args:
            dataset_path: Pasta raiz do dataset
            output_path: Onde salvar o JSON
            max_images_per_class: Número de imagens para samplear por classe
            
        Returns:
            Dicionário com {class_folder: description}
        """
        dataset_path = Path(dataset_path)
        
        # Encontra todas as subpastas (classes)
        class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        if not class_folders:
            print(f"⚠️  Nenhuma pasta de classe encontrada em {dataset_path}")
            return {}
        
        print(f"\n🎨 Processando dataset por classes...")
        print(f"   Classes encontradas: {len(class_folders)}")
        
        descriptors = {}
        failed = []
        
        for class_folder in tqdm(class_folders, desc="Processando classes"):
            try:
                class_name = class_folder.name
                
                # Coleta imagens da classe
                image_paths = []
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    image_paths.extend(class_folder.glob(f'*{ext}'))
                
                if not image_paths:
                    print(f"\n⚠️  Nenhuma imagem em {class_name}")
                    continue
                
                # Limita número de imagens se necessário
                if max_images_per_class:
                    image_paths = image_paths[:max_images_per_class]
                
                # Gera descrição para a PRIMEIRA imagem da classe (representativa)
                representative_image = str(image_paths[0])
                description = self.describe_image(representative_image)
                
                # Salva com o nome da CLASSE, não da imagem
                descriptors[class_name] = description
                
            except Exception as e:
                failed.append({"class": class_folder.name, "error": str(e)})
                print(f"\n❌ Erro em {class_folder.name}: {e}")
        
        # Salva JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Concluído!")
        print(f"   Classes processadas: {len(descriptors)}")
        print(f"   Falhas: {len(failed)}")
        print(f"   Salvo em: {output_path}")
        
        # Salva erros se houver
        if failed:
            error_path = output_path.replace('.json', '_errors.json')
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(failed, f, indent=2, ensure_ascii=False)
            print(f"   Erros em: {error_path}")
        
        return descriptors


def load_datasets_from_summary(summary_path: str) -> Dict[str, str]:
    """
    Carrega configuração de datasets a partir do summary.json
    
    Args:
        summary_path: Caminho para o arquivo summary.json
        
    Returns:
        Dicionário {nome_dataset: caminho_pasta}
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    datasets_config = {}
    
    # Se o JSON é uma lista de datasets
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            
            if dataset_name and dataset_path:
                datasets_config[dataset_name] = dataset_path
    
    # Se o JSON é um dicionário com chave "datasets"
    elif isinstance(summary, dict) and 'datasets' in summary:
        for item in summary['datasets']:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            
            if dataset_name and dataset_path:
                datasets_config[dataset_name] = dataset_path
    
    print(f"📊 Datasets carregados do summary: {len(datasets_config)}")
    for name in datasets_config.keys():
        print(f"   - {name}")
    
    return datasets_config


def get_concepts_for_dataset(dataset_name: str) -> List[str]:
    """Retorna conceitos apropriados para cada dataset"""
    
    concepts_map = {
        # Dogs
        "Stanford_Dogs": [
            "dog", "canine", "puppy", "pet", "domestic dog", "breed",
            "retriever", "labrador", "golden retriever", "german shepherd",
            "bulldog", "poodle", "beagle", "chihuahua", "husky", "corgi",
            "terrier", "spaniel", "setter", "pointer", "hound", "mastiff",
            "schnauzer", "boxer", "dalmatian", "dachshund", "pug",
            "shih tzu", "maltese", "pekingese", "japanese chin",
        ],
        
        # Birds
        "CUB_200_2011": [
            "bird", "avian", "songbird", "waterfowl", "raptor", "seabird",
            "warbler", "sparrow", "finch", "thrush", "wren", "robin",
            "cardinal", "jay", "crow", "raven", "blackbird", "oriole",
            "hummingbird", "woodpecker", "flycatcher", "swallow", "swift",
            "hawk", "eagle", "falcon", "owl", "vulture", "kite",
            "duck", "goose", "swan", "heron", "egret", "pelican",
            "gull", "tern", "sandpiper", "plover", "grebe",
        ],
        
        # Cars
        "Stanford_Cars": [
            "car", "automobile", "vehicle", "sedan", "coupe", "suv",
            "convertible", "hatchback", "wagon", "sports car", "luxury car",
            "compact car", "minivan", "pickup truck", "crossover",
            "bmw", "mercedes", "audi", "toyota", "honda", "ford",
        ],
        
        # Aircraft
        "FGVC_Aircraft": [
            "aircraft", "airplane", "plane", "jet", "airliner", "fighter",
            "boeing", "airbus", "cessna", "commercial aircraft", "military aircraft",
            "passenger plane", "cargo plane", "propeller plane", "biplane",
            "helicopter", "turboprop", "jumbo jet", "fighter jet",
        ],
        
        # Flowers
        "Oxford_Flowers_102": [
            "flower", "bloom", "blossom", "petal", "floral", "plant",
            "rose", "tulip", "daisy", "sunflower", "lily", "orchid",
            "carnation", "iris", "daffodil", "poppy", "peony", "dahlia",
            "hibiscus", "marigold", "chrysanthemum", "lavender",
        ],
        
        # Food
        "Food_101": [
            "food", "meal", "dish", "cuisine", "plate", "dessert", "snack",
            "pizza", "burger", "hamburger", "sushi", "pasta", "salad",
            "steak", "chicken", "fish", "soup", "sandwich", "taco",
            "cake", "ice cream", "pie", "cookie", "donut", "bread",
        ],
    }
    
    # Normaliza nome do dataset para matching
    dataset_normalized = dataset_name.replace("-", "_")
    
    # Tenta match exato primeiro
    if dataset_normalized in concepts_map:
        return concepts_map[dataset_normalized]
    
    # Tenta match parcial
    for key, concepts in concepts_map.items():
        if key.lower() in dataset_normalized.lower() or dataset_normalized.lower() in key.lower():
            return concepts
    
    # Fallback: conceitos genéricos
    print(f"⚠️  Dataset '{dataset_name}' não encontrado no mapa, usando conceitos genéricos")
    return [
        "object", "animal", "plant", "vehicle", "food", "tool",
        "bird", "dog", "cat", "fish", "insect", "flower", "tree",
        "car", "airplane", "building", "furniture", "item",
    ]


def generate_descriptors_for_datasets(
    datasets_config: Dict[str, str],
    output_dir: str = "descriptors",
    max_images_per_class: Optional[int] = 1,
    clip_model: str = "ViT-B/32"
):
    """
    Gera descriptors para múltiplos datasets (UM descriptor por CLASSE).
    
    Args:
        datasets_config: Dict com {nome_dataset: pasta_das_imagens}
        output_dir: Pasta onde salvar os JSONs
        max_images_per_class: Número de imagens por classe (1 = apenas representativa)
        clip_model: Modelo CLIP a usar
    """
    
    # Inicializa gerador
    generator = CLIPDescriptorGenerator(model_name=clip_model)
    generator.load_model()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Iniciando geração de descriptors POR CLASSE")
    print(f"📊 Datasets: {len(datasets_config)}")
    print(f"🔧 Modelo: {clip_model}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for dataset_name, dataset_path in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"📁 Dataset: {dataset_name}")
        
        # Prepara descrições específicas para o dataset
        concepts = get_concepts_for_dataset(dataset_name)
        generator.prepare_descriptions(concepts)
        
        # Processa dataset por classes
        output_path = os.path.join(output_dir, f"{dataset_name}_descriptors.json")
        generator.process_dataset_by_class(
            dataset_path=dataset_path,
            output_path=output_path,
            max_images_per_class=max_images_per_class
        )
    
    print(f"\n{'='*60}")
    print(f"✅ Todos os descriptors gerados!")
    print(f"📂 Verifique a pasta: {output_dir}")
    print(f"{'='*60}\n")


# ============== EXEMPLO DE USO ==============

if __name__ == "__main__":
    
    # Configuração usando o summary.json
    SUMMARY_PATH = Path("outputs/analysis/summary.json")
    OUTPUT_DIR = "descriptors"
    
    # Carrega datasets do summary
    datasets = load_datasets_from_summary(str(SUMMARY_PATH))
    
    # Gera descriptors (UM por CLASSE)
    generate_descriptors_for_datasets(
        datasets_config=datasets,
        output_dir=OUTPUT_DIR,
        max_images_per_class=1,  # Usa apenas 1 imagem representativa por classe
        clip_model="ViT-B/32"
    )
    
    print("\n🎉 Descriptors prontos para uso no zero-shot!")
    print("\n💡 Formato do JSON gerado:")
    print("   {")
    print('     "n02085620-Chihuahua": "a photo of a small dog",')
    print('     "n02085782-Japanese_spaniel": "a close-up photo of a dog"')
    print("   }")