"""
embedding_metrics.py
====================
Ergänzungs-Script für cluster_and_visualize_umap.py / tnse.py.
Läuft direkt auf den vorhandenen .pkl-Dateien — kein Umbau nötig.

Berechnet und plottet:
1. Silhouette Score pro Modell (wie gut sind Prompt-Cluster getrennt?)
2. Inter-Modell Kosinus-Distanz Matrix (West vs. Ost pro Prompt)
3. Kompakte "Key Numbers" Ausgabe für die Fazit-Folie

Verwendung:
    python embedding_metrics.py            # Standard: CLIP
    python embedding_metrics.py --type clip
    python embedding_metrics.py --type dinov3
"""

import argparse
import glob
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# =============================================================
# PFADE (identisch zu deinen anderen Scripts)
# =============================================================
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != "analysis" else SCRIPT_DIR.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
PLOTS_OUT    = OUTPUT_DIR / "plots" / "embedding_metrics"

# =============================================================
# HILFSFUNKTIONEN
# =============================================================
def load_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_all_pkls(feature_type: str) -> dict[str, list]:
    """Lädt alle PKL-Dateien für den gewählten Feature-Typ."""
    pattern   = str(OUTPUT_DIR / f"{feature_type}_embeddings_*.pkl")
    pkl_files = glob.glob(pattern)
    if not pkl_files:
        raise FileNotFoundError(
            f"Keine PKL-Dateien gefunden für Muster: {pattern}\n"
            f"Bitte zuerst extract_clip_embeddings.py oder extract_dinov3_embeddings.py ausführen."
        )
    result = {}
    for f in pkl_files:
        match = re.search(rf"{feature_type}_embeddings_(.+)\.pkl", Path(f).name)
        if match:
            model_name = match.group(1)
            result[model_name] = load_pkl(f)
            print(f"  ✅ Geladen: {Path(f).name}  ({len(result[model_name])} Bilder)")
    return result

def pkls_to_dataframe(models_data: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Konvertiert alle PKL-Daten in eine gemeinsame Embedding-Matrix + Metadaten-DF."""
    rows        = []
    embeddings  = []
    for model_name, items in models_data.items():
        for item in items:
            rows.append({
                "Image_Name":     item["Image_Name"],
                "T2I_Model":      model_name.lower().strip(),
                "Prompt_Subject": item.get("Prompt_Subject", "unknown"),
            })
            embeddings.append(item["Embedding"])
    df = pd.DataFrame(rows)
    X  = np.array(embeddings)
    return X, df

# =============================================================
# METRIK 1: SILHOUETTE SCORE PRO MODELL
# =============================================================
def compute_silhouette_per_model(X: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet den Silhouette Score pro T2I-Modell.
    Fragestellung: Wie gut trennt das Modell seine Prompts im Vektorraum?
    Score nahe 1.0 = Prompts sind klar separiert (starke Bias-Muster).
    Score nahe 0.0 = Prompts vermischen sich (kein klares Muster).
    """
    results = []
    models  = df["T2I_Model"].unique()

    for model in sorted(models):
        mask    = df["T2I_Model"] == model
        X_model = X[mask]
        labels  = df.loc[mask, "Prompt_Subject"].values

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"  ⚠️ {model}: Nur {len(unique_labels)} Prompt-Klasse(n) — Silhouette nicht berechenbar.")
            continue
        if len(X_model) < len(unique_labels) + 1:
            print(f"  ⚠️ {model}: Zu wenige Bilder ({len(X_model)}) für {len(unique_labels)} Klassen.")
            continue

        le            = LabelEncoder()
        labels_enc    = le.fit_transform(labels)
        score         = silhouette_score(X_model, labels_enc, metric="cosine")

        results.append({
            "Modell":           model,
            "Silhouette_Score": round(float(score), 4),
            "N_Bilder":         int(mask.sum()),
            "N_Prompts":        int(len(unique_labels)),
            "Interpretation":   (
                "stark getrennt"  if score > 0.5 else
                "mittel getrennt" if score > 0.25 else
                "schwach getrennt"
            )
        })
        print(f"  📐 {model:20s}  Silhouette = {score:.4f}  ({results[-1]['Interpretation']})")

    return pd.DataFrame(results)


# =============================================================
# METRIK 2: INTER-MODELL KOSINUS-DISTANZ PRO PROMPT
# =============================================================
def compute_inter_model_cosine(X: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet für jeden Prompt den durchschnittlichen Kosinus-Abstand
    zwischen je zwei Modellen (paarweise).
    Niedriger Wert (~0.0) = Modelle erzeugen für diesen Prompt ähnliche Bilder.
    Hoher Wert (~1.0)     = Modelle erzeugen grundlegend verschiedene Bilder.
    """
    prompts = sorted(df["Prompt_Subject"].dropna().unique())
    models  = sorted(df["T2I_Model"].unique())
    rows    = []

    for prompt in prompts:
        # Durchschnitts-Vektor pro Modell für diesen Prompt
        model_vectors = {}
        for model in models:
            mask = (df["T2I_Model"] == model) & (df["Prompt_Subject"] == prompt)
            if mask.sum() == 0:
                continue
            model_vectors[model] = X[mask].mean(axis=0)

        model_names = list(model_vectors.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                mA, mB = model_names[i], model_names[j]
                vA = model_vectors[mA].reshape(1, -1)
                vB = model_vectors[mB].reshape(1, -1)
                sim  = cosine_similarity(vA, vB)[0][0]
                dist = 1.0 - sim  # Kosinus-Distanz (0 = identisch, 1 = maximal verschieden)
                rows.append({
                    "Prompt":           prompt,
                    "Modell_A":         mA,
                    "Modell_B":         mB,
                    "Kosinus_Distanz":  round(float(dist), 4),
                    "Kosinus_Aehnlichkeit": round(float(sim), 4),
                })

    return pd.DataFrame(rows)


# =============================================================
# PLOTS
# =============================================================
def plot_silhouette(df_sil: pd.DataFrame, feature_type: str, out_dir: Path):
    """Horizontales Balkendiagramm der Silhouette Scores."""
    if df_sil.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2ca02c" if s > 0.5 else ("#ff7f0e" if s > 0.25 else "#d62728")
               for s in df_sil["Silhouette_Score"]]

    bars = ax.barh(df_sil["Modell"], df_sil["Silhouette_Score"],
                   color=colors, edgecolor="black", linewidth=0.6)

    # Werte beschriften
    for bar, val in zip(bars, df_sil["Silhouette_Score"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11)

    ax.axvline(x=0.5,  color="black", linestyle="--", linewidth=1.2, label="Gut (0.5)")
    ax.axvline(x=0.25, color="gray",  linestyle=":",  linewidth=1.0, label="Mittel (0.25)")
    ax.set_xlim(0, max(df_sil["Silhouette_Score"].max() + 0.15, 0.6))
    ax.set_xlabel("Silhouette Score (Kosinus-Distanz)\n← schlecht  |  gut →")
    ax.set_title(
        f"Silhouette Score pro Modell [{feature_type.upper()}]\n"
        f"Wie stark trennt das Modell seine Prompt-Themen im Vektorraum?"
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / f"SILHOUETTE_SCORE_{feature_type.upper()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  💾 {path.name}")


def plot_inter_model_heatmap(df_dist: pd.DataFrame, feature_type: str, out_dir: Path):
    """
    Heatmap-Matrix: Für jeden Prompt eine Zeile, für jedes Modell-Paar eine Spalte.
    Zeigt auf einen Blick wo West- und Ost-Modelle am stärksten divergieren.
    """
    if df_dist.empty:
        return

    df_dist["Pair"] = df_dist["Modell_A"] + " vs " + df_dist["Modell_B"]
    pivot = df_dist.pivot_table(
        index="Prompt", columns="Pair", values="Kosinus_Distanz", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2.2), max(6, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
        vmin=0.0, vmax=0.5,
        cbar_kws={"label": "Kosinus-Distanz (0=identisch, 1=maximal verschieden)"}
    )
    ax.set_title(
        f"Inter-Modell Kosinus-Distanz pro Prompt [{feature_type.upper()}]\n"
        f"Hohe Werte = Modelle erzeugen grundlegend verschiedene Bilder für diesen Prompt"
    )
    ax.set_xlabel("Modell-Paar")
    ax.set_ylabel("Prompt-Thema")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = out_dir / f"INTER_MODEL_COSINE_DISTANCE_{feature_type.upper()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  💾 {path.name}")


def plot_distance_by_prompt(df_dist: pd.DataFrame, feature_type: str, out_dir: Path):
    """
    Grouped Bar Chart: Für jedes Modell-Paar die Distanz pro Prompt.
    Zeigt welche Prompts (CEO, Nurse, ...) den größten West-Ost-Split erzeugen.
    """
    if df_dist.empty:
        return

    df_dist["Pair"] = df_dist["Modell_A"] + " vs\n" + df_dist["Modell_B"]
    pairs = df_dist["Pair"].unique()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df_dist, x="Prompt", y="Kosinus_Distanz", hue="Pair",
        palette="Set2", ax=ax
    )
    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=1.0, label="Referenz (0.1)")
    ax.set_xlabel("Prompt-Thema")
    ax.set_ylabel("Ø Kosinus-Distanz zwischen den Modellen")
    ax.set_title(
        f"Welcher Prompt erzeugt den größten West-Ost-Split? [{feature_type.upper()}]\n"
        f"Höhere Balken = Modelle weichen bei diesem Thema stärker voneinander ab"
    )
    ax.legend(title="Modell-Paar", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / f"COSINE_DISTANCE_BY_PROMPT_{feature_type.upper()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  💾 {path.name}")


# =============================================================
# KONSOLEN-AUSGABE: KEY NUMBERS FÜR FAZIT-FOLIE
# =============================================================
def print_key_numbers(df_sil: pd.DataFrame, df_dist: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  📊 KEY NUMBERS — direkt für Fazit-Folie verwendbar")
    print("═" * 60)

    if not df_sil.empty:
        print("\n  SILHOUETTE SCORES (Prompt-Trennung im Vektorraum):")
        for _, r in df_sil.sort_values("Silhouette_Score", ascending=False).iterrows():
            bar = "█" * int(r["Silhouette_Score"] * 20)
            print(f"    {r['Modell']:20s}  {r['Silhouette_Score']:.4f}  {bar}  → {r['Interpretation']}")

    if not df_dist.empty:
        print("\n  GRÖSSTE INTER-MODELL DISTANZEN (Top 5 Prompt-Paar-Kombinationen):")
        top5 = df_dist.nlargest(5, "Kosinus_Distanz")
        for _, r in top5.iterrows():
            print(f"    {r['Modell_A']:12s} vs {r['Modell_B']:12s}  |  "
                  f"Prompt: {r['Prompt']:20s}  |  Distanz: {r['Kosinus_Distanz']:.4f}")

        print("\n  KLEINSTE INTER-MODELL DISTANZEN — wo sind Modelle am ähnlichsten? (Top 5):")
        bot5 = df_dist.nsmallest(5, "Kosinus_Distanz")
        for _, r in bot5.iterrows():
            print(f"    {r['Modell_A']:12s} vs {r['Modell_B']:12s}  |  "
                  f"Prompt: {r['Prompt']:20s}  |  Distanz: {r['Kosinus_Distanz']:.4f}")

    print("\n" + "═" * 60)


# =============================================================
# MAIN
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Embedding Metrics: Silhouette + Inter-Modell Kosinus-Distanz")
    parser.add_argument("--type", choices=["clip", "dinov3"], default="clip",
                        help="Welcher Embedding-Typ? (default: clip)")
    args = parser.parse_args()

    feature_type = args.type

    print("=" * 60)
    print(f"📐 EMBEDDING METRICS  [{feature_type.upper()}]")
    print("=" * 60)

    PLOTS_OUT.mkdir(parents=True, exist_ok=True)

    # --- Daten laden ---
    print("\n[1/4] PKL-Dateien laden...")
    try:
        models_data = load_all_pkls(feature_type)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print(f"\n       {len(models_data)} Modelle geladen: {list(models_data.keys())}")

    # --- Gemeinsame Matrix bauen ---
    print("\n[2/4] Embedding-Matrix zusammenbauen...")
    X, df = pkls_to_dataframe(models_data)
    print(f"       {len(X)} Bilder total  |  {X.shape[1]}-dimensionale Vektoren")

    # --- Metrik 1: Silhouette ---
    print("\n[3/4] Silhouette Score berechnen (kann ~30 Sek. dauern)...")
    df_sil = compute_silhouette_per_model(X, df)

    # --- Metrik 2: Inter-Modell Distanz ---
    print("\n[4/4] Inter-Modell Kosinus-Distanz berechnen...")
    df_dist = compute_inter_model_cosine(X, df)

    # --- CSVs speichern ---
    if not df_sil.empty:
        p = PLOTS_OUT / f"SILHOUETTE_{feature_type.upper()}.csv"
        df_sil.to_csv(p, index=False)
        print(f"\n  💾 {p.name}")

    if not df_dist.empty:
        p = PLOTS_OUT / f"INTER_MODEL_DISTANCES_{feature_type.upper()}.csv"
        df_dist.to_csv(p, index=False)
        print(f"  💾 {p.name}")

    # --- Plots ---
    print("\n  Plots werden generiert...")
    plot_silhouette(df_sil, feature_type, PLOTS_OUT)
    plot_inter_model_heatmap(df_dist, feature_type, PLOTS_OUT)
    plot_distance_by_prompt(df_dist, feature_type, PLOTS_OUT)

    # --- Key Numbers ---
    print_key_numbers(df_sil, df_dist)

    print(f"\n✅ Fertig! Alle Ergebnisse in: {PLOTS_OUT}")


if __name__ == "__main__":
    main()