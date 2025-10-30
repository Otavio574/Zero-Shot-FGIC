# scripts/summarize_zero_shot_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def summarize_results(path="zero_shot_results.json"):
    path = Path(path)
    if not path.exists():
        print(f"❌ Arquivo {path} não encontrado.")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    df = pd.DataFrame([
        {
            "dataset": name,
            "accuracy": r["accuracy"],
            "num_classes": r["num_classes"],
            "num_images": r["num_images"],
        }
        for name, r in results.items()
    ])

    df = df.sort_values(by="accuracy", ascending=False)
    print("\n📊 Resultados Zero-Shot — CLIP Baseline:\n")
    print(df.to_string(index=False))

    print("\n📈 Estatísticas globais:")
    print(f"Média das acurácias: {df['accuracy'].mean():.4f}")
    print(f"Mediana: {df['accuracy'].median():.4f}")
    print(f"Desvio padrão: {df['accuracy'].std():.4f}")

    # Salvar CSV
    csv_path = path.with_name("zero_shot_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Resumo salvo em: {csv_path}")

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.barh(df["dataset"], df["accuracy"], color="royalblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Dataset")
    plt.title(f"Zero-Shot Accuracy — {data['model']}")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    img_path = path.with_name("zero_shot_summary.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    print(f"📊 Gráfico salvo em: {img_path}")

    plt.show()

if __name__ == "__main__":
    summarize_results()
