# src/main.py
import argparse
import os
from src.load_data import list_image_paths, build_class_names
from src.generate_prompts import make_prompts_for_all, DEFAULT_TEMPLATES
from src.inference_clip import CLIPZeroShot, device
from src.evaluate import compute_topk_from_similarities, save_predictions_csv
from tqdm import tqdm

def run_pipeline(dataset_root: str, out_csv: str, model_name: str = "openai/clip-vit-base-patch32",
                 batch_size: int = 32, templates = None):
    print(f"[INFO] Dataset root: {dataset_root}")
    items = list_image_paths(dataset_root)
    if len(items) == 0:
        raise ValueError("Nenhuma imagem encontrada. Verifique o dataset_root e a estrutura.")
    image_paths = [p for p, c in items]
    true_classes = [c for p, c in items]
    class_names = build_class_names(dataset_root)
    print(f"[INFO] Found {len(class_names)} classes and {len(image_paths)} images")

    if templates is None:
        templates = DEFAULT_TEMPLATES

    prompts = make_prompts_for_all(class_names, templates)

    clip_model = CLIPZeroShot(model_name=model_name)
    class_names_ordered, text_embs = clip_model.text_embeddings_mean(prompts)
    # Guarantee same ordering
    assert class_names_ordered == class_names, "Ordering mismatch between class names."

    # For memory safety, predict in batches using generator
    print(f"[INFO] Running inference on {len(image_paths)} images (batch_size={batch_size})")
    similarities = clip_model.predict(image_paths, class_names_ordered, text_embs, batch_size=batch_size)

    # Evaluate
    results = compute_topk_from_similarities(similarities, class_names_ordered, true_classes, topk=(1,5))
    print(f"[RESULT] Top-1: {results['top1']:.4f}, Top-5: {results['top5']:.4f}")

    # Save outputs: predictions CSV (with top1, top5, true)
    df = results["df"]
    save_predictions_csv(df, out_csv)
    print(f"[INFO] Predictions saved to {out_csv}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path root dataset (class subfolders)")
    parser.add_argument("--out", type=str, default="predictions.csv")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run_pipeline(args.dataset, args.out, model_name=args.model, batch_size=args.batch)
