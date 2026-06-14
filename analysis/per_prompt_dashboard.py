"""
per_prompt_dashboard.py
=======================
Erzeugt pro VLM-Modell × T2I-Modell × Prompt einen 3-Panel Plot:
    [Gender]  [Ethnicity]  [Skin Tone (MST)]
Einmal für MACRO (alle Bilder) und einmal für FAIR (Human-Eval Subset).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Farbpaletten konsistent mit evaluate_results.py
GENDER_ORDER    = ["Man", "Woman", "Unclear"]
ETHNICITY_ORDER = ["White", "Asian", "Black", "Indian", "Latino Hispanic", "Middle Eastern", "Unclear"]
MST_ORDER       = [str(i) for i in range(1, 11)] + ["Unclear"]

GENDER_COLORS     = {"Man": "#5b9bd5", "Woman": "#ed7d31", "Unclear": "#a5a5a5"}
ETHNICITY_PALETTE = "Set2"
MST_PALETTE       = "rocket_r"

# Die VLMs, die wir auswerten wollen
VLM_LIST = ['gemma4:e4b', 'qwen2.5vl:7b', 'blaifa/InternVL3_5:8B']

def _find_skin_col(df: pd.DataFrame) -> str | None:
    for candidate in ['ITA_Scale_MST', 'MonkScale_RGB', 'VLM_MST']:
        if candidate in df.columns:
            return candidate
    mst_cols = [c for c in df.columns if 'MST' in c or 'Skin' in c.title()]
    return mst_cols[0] if mst_cols else None

# =============================================================
# KERN: EIN DASHBOARD PRO MODELL + PROMPT + VLM
# =============================================================
def _build_dashboard(
    df_prompt: pd.DataFrame,
    model_name: str,
    prompt_name: str,
    dataset_label: str,
    output_folder: Path,
    vlm_gender_col: str | None,
    vlm_race_col:   str | None,
    skin_col:       str | None,
    vlm_filename_tag: str
):
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"[{dataset_label}]  Modell: {model_name.upper()}  |  Prompt: {prompt_name}  "
        f"(n={len(df_prompt)} Bilder)",
        fontsize=13, fontweight='bold', y=1.01
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

    # ── Panel 1: GENDER ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    if vlm_gender_col and vlm_gender_col in df_prompt.columns:
        counts = df_prompt[vlm_gender_col].value_counts().reindex(GENDER_ORDER, fill_value=0)
        total  = counts.sum()
        pcts   = (counts / total * 100).round(1) if total > 0 else counts * 0

        bars = ax1.bar(
            counts.index, counts.values,
            color=[GENDER_COLORS.get(g, "#cccccc") for g in counts.index],
            edgecolor="black", linewidth=0.5
        )
        for bar, pct in zip(bars, pcts.values):
            if bar.get_height() > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
                )
        ax1.axhline(y=total / 2, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="50% Parität")
        ax1.set_title("Gender", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Anzahl Bilder")
        ax1.set_ylim(0, max(counts.values) * 1.2 + 1)
        ax1.legend(fontsize=8, loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Keine Gender-Daten", ha="center", va="center", transform=ax1.transAxes, color="gray")
        ax1.set_title("Gender", fontsize=11, fontweight="bold")

    # ── Panel 2: ETHNICITY ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if vlm_race_col and vlm_race_col in df_prompt.columns:
        counts_eth = df_prompt[vlm_race_col].value_counts()
        ordered = [e for e in ETHNICITY_ORDER if e in counts_eth.index]
        rest    = [e for e in counts_eth.index  if e not in ETHNICITY_ORDER]
        counts_eth = counts_eth.reindex(ordered + rest, fill_value=0)
        total_eth  = counts_eth.sum()
        pcts_eth   = (counts_eth / total_eth * 100).round(1) if total_eth > 0 else counts_eth * 0

        palette = sns.color_palette(ETHNICITY_PALETTE, n_colors=len(counts_eth))
        bars2   = ax2.bar(
            range(len(counts_eth)), counts_eth.values,
            color=palette, edgecolor="black", linewidth=0.5
        )
        for bar, pct in zip(bars2, pcts_eth.values):
            if bar.get_height() > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8
                )
        ax2.set_xticks(range(len(counts_eth)))
        ax2.set_xticklabels(counts_eth.index, rotation=35, ha="right", fontsize=8)
        ax2.set_title("Ethnicity", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Anzahl Bilder")
        ax2.set_ylim(0, max(counts_eth.values) * 1.2 + 1)
    else:
        ax2.text(0.5, 0.5, "Keine Ethnicity-Daten", ha="center", va="center", transform=ax2.transAxes, color="gray")
        ax2.set_title("Ethnicity", fontsize=11, fontweight="bold")

    # ── Panel 3: SKIN TONE (MST) ─────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if skin_col and skin_col in df_prompt.columns:
        skin_data  = df_prompt[skin_col].astype(str)
        counts_mst = skin_data.value_counts()
        ordered_mst = [s for s in MST_ORDER if s in counts_mst.index]
        rest_mst    = [s for s in counts_mst.index if s not in MST_ORDER]
        counts_mst  = counts_mst.reindex(ordered_mst + rest_mst, fill_value=0)
        total_mst   = counts_mst.sum()
        pcts_mst    = (counts_mst / total_mst * 100).round(1) if total_mst > 0 else counts_mst * 0

        n = len(counts_mst)
        gradient = plt.cm.YlOrBr(np.linspace(0.15, 0.95, n))
        bars3 = ax3.bar(
            range(n), counts_mst.values,
            color=gradient, edgecolor="black", linewidth=0.5
        )
        for bar, pct in zip(bars3, pcts_mst.values):
            if bar.get_height() > 0:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8
                )
        ax3.set_xticks(range(n))
        ax3.set_xticklabels(counts_mst.index, rotation=35, ha="right", fontsize=8)
        ax3.set_title(f"Skin Tone ({skin_col})", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Anzahl Bilder")
        ax3.set_ylim(0, max(counts_mst.values) * 1.2 + 1)
    else:
        ax3.text(0.5, 0.5, "Keine Skin-Tone-Daten", ha="center", va="center", transform=ax3.transAxes, color="gray")
        ax3.set_title("Skin Tone (MST)", fontsize=11, fontweight="bold")

    # Speichern (mit VLM Name im Dateinamen!)
    safe_prompt = prompt_name.replace("/", "_").replace(" ", "_")
    clean_dataset = dataset_label.split(" |")[0] # Entfernt den VLM-Zusatz für den Dateinamen
    filename    = f"DASHBOARD_{clean_dataset}_{model_name}_{safe_prompt}_{vlm_filename_tag}.png"
    
    plt.tight_layout()
    plt.savefig(output_folder / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# HAUPT-EINSTIEGSPUNKT
# =============================================================
def generate_per_prompt_dashboards(
    raw_dataframes: dict,
    model_name: str,
    output_folder: Path,
    dataset_label: str = "MACRO",
    is_fair: bool = False,
    master_df: pd.DataFrame | None = None,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    if is_fair and master_df is not None:
        # FAIR-MODUS: master_df hat pivotierte Spalten (z.B. gemma4:e4b_VLM_Gender)
        df_combined = master_df.copy()
        prompts = sorted(df_combined["Prompt_Subject"].dropna().unique())
        skin_col = _find_skin_col(df_combined)

        for vlm in VLM_LIST:
            vlm_safe_name = vlm.split('/')[-1].replace(':', '_')
            
            # Suche die spezifischen Spalten für GENAU dieses VLM
            vlm_gender_col = next((c for c in df_combined.columns if vlm in c and 'Gender' in c), None)
            vlm_race_col   = next((c for c in df_combined.columns if vlm in c and 'Race' in c), None)

            if not vlm_gender_col and not vlm_race_col:
                continue

            for prompt in prompts:
                df_prompt = df_combined[df_combined["Prompt_Subject"] == prompt].copy()
                if df_prompt.empty: continue

                _build_dashboard(
                    df_prompt      = df_prompt,
                    model_name     = model_name,
                    prompt_name    = prompt,
                    dataset_label  = f"{dataset_label} | VLM: {vlm_safe_name}",
                    output_folder  = output_folder,
                    vlm_gender_col = vlm_gender_col,
                    vlm_race_col   = vlm_race_col,
                    skin_col       = skin_col,
                    vlm_filename_tag = vlm_safe_name
                )
        print(f"  [{dataset_label}] Dashboards für alle VLMs gespeichert → {output_folder}")

    else:
        # MACRO-MODUS: raw_dataframes
        df_ollama = raw_dataframes.get("Ollama")
        df_skin   = raw_dataframes.get("Skin")

        if df_ollama is None or df_ollama.empty:
            return

        for vlm in VLM_LIST:
            vlm_safe_name = vlm.split('/')[-1].replace(':', '_')

            # Filtere die OLLAMA-Daten streng auf das aktuelle VLM!
            df_base = df_ollama[
                (df_ollama["T2I_Model"] == model_name) &
                (df_ollama["VLM_Model"] == vlm)
            ].copy()

            if df_base.empty:
                continue

            # Skin Tone wieder anheften
            if df_skin is not None and not df_skin.empty:
                df_skin_model = df_skin[df_skin["T2I_Model"] == model_name][
                    ["Image_Name", "T2I_Model", "ITA_Scale_MST", "MonkScale_RGB"]
                ]
                df_combined = pd.merge(df_base, df_skin_model, on=["Image_Name", "T2I_Model"], how="left")
            else:
                df_combined = df_base

            vlm_gender_col = "VLM_Gender" if "VLM_Gender" in df_combined.columns else None
            vlm_race_col   = "VLM_Race"   if "VLM_Race"   in df_combined.columns else None
            skin_col       = _find_skin_col(df_combined)

            if "Prompt_Subject" not in df_combined.columns:
                continue

            prompts = sorted(df_combined["Prompt_Subject"].dropna().unique())

            for prompt in prompts:
                df_prompt = df_combined[df_combined["Prompt_Subject"] == prompt].copy()
                if df_prompt.empty: continue

                _build_dashboard(
                    df_prompt      = df_prompt,
                    model_name     = model_name,
                    prompt_name    = prompt,
                    dataset_label  = f"{dataset_label} | VLM: {vlm_safe_name}",
                    output_folder  = output_folder,
                    vlm_gender_col = vlm_gender_col,
                    vlm_race_col   = vlm_race_col,
                    skin_col       = skin_col,
                    vlm_filename_tag = vlm_safe_name
                )
        print(f"  [{dataset_label}] Dashboards für alle VLMs gespeichert → {output_folder}")