# src/inference_clip.py
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPZeroShot:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = device):
        self.device = device
        print(f"[INFO] Loading CLIP model {model_name} on {device}...")
        # carregar modelo
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.to(self.device) #type: ignore
        self.model.eval()
        # carregar processor
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    def text_embeddings_mean(self, prompts_per_class: Dict[str, List[str]]) -> Tuple[List[str], torch.Tensor]:
        class_names = list(prompts_per_class.keys())
        all_embs = []
        for c in tqdm(class_names, desc="Text embeddings"):
            texts = prompts_per_class[c]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                txt_feats = self.model.get_text_features(**inputs)
            txt_feats = self._normalize(txt_feats)
            mean_emb = txt_feats.mean(dim=0, keepdim=True)
            mean_emb = self._normalize(mean_emb)
            all_embs.append(mean_emb.squeeze(0))
        text_embs = torch.stack(all_embs, dim=0)
        return class_names, text_embs

    def image_batch_embeddings(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=imgs, return_tensors="pt")  # type: ignore
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img_feats = self.model.get_image_features(**inputs)
        img_feats = self._normalize(img_feats)
        return img_feats

    def image_batch_embeddings_stream(self, image_paths: List[str], batch_size: int = 32):
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i+batch_size]
            yield self.image_batch_embeddings(chunk, batch_size=batch_size)

    def predict(self, image_paths: List[str], class_names: List[str], text_embs: torch.Tensor, batch_size: int = 32):
        results = []
        text_embs = text_embs.to(self.device)
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt")  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
            img_feats = self._normalize(img_feats)
            sims = img_feats @ text_embs.T
            sims = sims.cpu().numpy()
            for row in sims:
                results.append(row)
        return results
