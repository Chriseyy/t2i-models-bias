"""
per_prompt_statistics.py
========================
Berechnet für jede Kombination Modell × Prompt × Dimension (Gender + Ethnicity):
  - Rf(p)          : Frauenanteil (Gender, 0.5 = fair)
  - Chi²           : Signifikanztest gegen faire Gleichverteilung
  - Entropy        : Diversitätsmaß der Verteilung (hoch = divers, niedrig = dominiert)
  - Dominanz       : Häufigste Kategorie + ihr Anteil in %
  - Absolute Counts: Man / Woman / Unclear  bzw. jede Ethnie

NEU: Generiert getrennte Übersichts-Plots für Gender und Ethnie,
um die unterschiedlichen Chi²-Skalen methodisch sauber zu trennen!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import entropy as scipy_entropy

# ── Konstanten ────────────────────────────────────────────────
GENDER_CATS    = ["Man", "Woman", "Unclear"]
ETHNICITY_CATS = ["White", "Asian", "Black", "Indian",
                  "Latino Hispanic", "Middle Eastern", "Unclear"]

CHI2_CRIT_GENDER    = 3.841   # df=1
CHI2_CRIT_ETHNICITY = 12.592  # df=6

# Die VLMs, die wir separat auswerten wollen
VLM_LIST = ['gemma4:e4b', 'qwen2.5vl:7b', 'blaifa/InternVL3_5:8B']


# =============================================================
# HILFSFUNKTIONEN FÜR STATISTIKEN
# =============================================================
def _rf(series: pd.Series) -> float:
    female = (series == "Woman").sum()
    male   = (series == "Man").sum()
    total  = female + male
    return round(float(female / total), 4) if total > 0 else np.nan

def _chi2_gender(series: pd.Series) -> tuple[float, str]:
    male   = (series == "Man").sum()
    female = (series == "Woman").sum()
    total  = male + female
    if total < 2: return np.nan, "n/a"
    expected = total / 2
    chi2 = ((male - expected) ** 2 / expected) + ((female - expected) ** 2 / expected)
    sig  = "p<0.05" if chi2 > CHI2_CRIT_GENDER else "n.s."
    return round(float(chi2), 3), sig

def _chi2_ethnicity(series: pd.Series) -> tuple[float, str]:
    counts = series[series != "Unclear"].value_counts()
    if len(counts) < 2: return np.nan, "n/a"
    total    = counts.sum()
    expected = total / len(counts)
    chi2_val = sum((o - expected) ** 2 / expected for o in counts.values)
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2_val, df=len(counts) - 1)
    sig   = "p<0.05" if p_val < 0.05 else "n.s."
    return round(float(chi2_val), 3), sig

def _entropy_norm(series: pd.Series, n_cats: int) -> float:
    counts = series.dropna().value_counts().values
    if counts.sum() == 0 or len(counts) < 2: return 0.0
    probs    = counts / counts.sum()
    h        = scipy_entropy(probs, base=2)
    h_max    = np.log2(n_cats)
    return round(float(h / h_max), 4) if h_max > 0 else 0.0

def _dominant(series: pd.Series) -> tuple[str, float]:
    vc = series.dropna().value_counts()
    if vc.empty: return "N/A", np.nan
    top = vc.index[0]
    pct = round(float(vc.iloc[0] / vc.sum() * 100), 1)
    return top, pct


# =============================================================
# METRIKEN BERECHNEN (NUR FÜR DAS AKTUELLE VLM)
# =============================================================
def compute_per_prompt_metrics(master_df: pd.DataFrame, vlm_name: str) -> pd.DataFrame:
    # Finde die exakten Spaltennamen im Master-DF für dieses spezielle VLM
    gender_col = next((c for c in master_df.columns if vlm_name in c and 'Gender' in c), None)
    race_col   = next((c for c in master_df.columns if vlm_name in c and 'Race' in c), None)

    rows = []
    
    if "T2I_Model" not in master_df.columns or "Prompt_Subject" not in master_df.columns:
        return pd.DataFrame()

    for model in sorted(master_df["T2I_Model"].dropna().unique()):
        df_m = master_df[master_df["T2I_Model"] == model]

        for prompt in sorted(df_m["Prompt_Subject"].dropna().unique()):
            df_p = df_m[df_m["Prompt_Subject"] == prompt]
            n    = len(df_p)

            base = {
                "Modell":  model,
                "Prompt":  prompt,
                "N_Bilder": n,
            }

            # ── GENDER ────────────────────────────────────────
            if gender_col and gender_col in df_p.columns:
                g_series = df_p[gender_col].astype(str).str.title().str.strip()
                rf_val          = _rf(g_series)
                chi2_g, sig_g   = _chi2_gender(g_series)
                ent_g           = _entropy_norm(g_series, n_cats=len(GENDER_CATS))
                dom_g, dom_pct_g = _dominant(g_series[g_series != "Unclear"])

                vc_g = g_series.value_counts()
                row_g = {**base,
                    "Dimension":        "Gender",
                    "Rf_p":             rf_val,
                    "Chi2":             chi2_g,
                    "Chi2_Signifikanz": sig_g,
                    "Entropy_norm":     ent_g,
                    "Dominant_Cat":     dom_g,
                    "Dominant_Pct":     dom_pct_g,
                    "Man":              int(vc_g.get("Man",     0)),
                    "Woman":            int(vc_g.get("Woman",   0)),
                    "Unclear":          int(vc_g.get("Unclear", 0)),
                }
                rows.append(row_g)

            # ── ETHNICITY ─────────────────────────────────────
            if race_col and race_col in df_p.columns:
                r_series = df_p[race_col].astype(str).str.title().str.strip()
                chi2_r, sig_r    = _chi2_ethnicity(r_series)
                ent_r            = _entropy_norm(r_series, n_cats=len(ETHNICITY_CATS))
                dom_r, dom_pct_r = _dominant(r_series[r_series != "Unclear"])

                vc_r = r_series.value_counts()
                row_r = {**base,
                    "Dimension":        "Ethnicity",
                    "Rf_p":             np.nan,
                    "Chi2":             chi2_r,
                    "Chi2_Signifikanz": sig_r,
                    "Entropy_norm":     ent_r,
                    "Dominant_Cat":     dom_r,
                    "Dominant_Pct":     dom_pct_r,
                }
                for cat in ETHNICITY_CATS:
                    row_r[cat] = int(vc_r.get(cat, 0))
                rows.append(row_r)

    return pd.DataFrame(rows)


# =============================================================
# PLOT-GENERATOREN
# =============================================================
def plot_chi2_heatmaps(df_stats: pd.DataFrame, output_folder: Path, label_title: str, file_tag: str):
    for dim, crit, title_suffix in [
        ("Gender",    CHI2_CRIT_GENDER,    "Gender  (H₀: 50% Man / 50% Woman)"),
        ("Ethnicity", CHI2_CRIT_ETHNICITY, "Ethnicity  (H₀: Gleichverteilung)"),
    ]:
        sub = df_stats[df_stats["Dimension"] == dim].copy()
        if sub.empty: continue

        pivot = sub.pivot_table(index="Modell", columns="Prompt", values="Chi2", aggfunc="mean")
        sig_pivot = sub.pivot_table(index="Modell", columns="Prompt", values="Chi2_Signifikanz", aggfunc="first")

        if pivot.empty: continue

        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.8), max(5, len(pivot) * 1.2)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax, cbar_kws={"label": f"Chi²-Wert  (Signifikanzgrenze: {crit})"})

        for r_idx, model in enumerate(pivot.index):
            for c_idx, prompt in enumerate(pivot.columns):
                try:
                    if sig_pivot.loc[model, prompt] == "p<0.05":
                        ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor="red", lw=2.5))
                except KeyError: pass

        ax.set_title(f"[{label_title}] Chi²-Heatmap: {title_suffix}\nRoter Rahmen = signifikant (p<0.05)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prompt-Thema")
        ax.set_ylabel("T2I-Modell")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        fname = output_folder / f"CHI2_HEATMAP_{dim.upper()}_{file_tag}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

def plot_per_model_overview_combined(df_stats: pd.DataFrame, output_folder: Path, label_title: str, file_tag: str):
    """Der alte, kombinierte Plot (Gender + Ethnie nebeneinander)"""
    models = df_stats["Modell"].dropna().unique()

    for model in sorted(models):
        df_m = df_stats[df_stats["Modell"] == model].copy()
        prompts = sorted(df_m["Prompt"].dropna().unique())
        x = np.arange(len(prompts))

        df_g = df_m[df_m["Dimension"] == "Gender"].set_index("Prompt").reindex(prompts)
        df_e = df_m[df_m["Dimension"] == "Ethnicity"].set_index("Prompt").reindex(prompts)

        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(prompts) * 1.6), 13), sharex=True)
        fig.suptitle(f"[{label_title}]  Modell: {model.upper()}  —  Alle Prompts (Kombiniert)", fontsize=13, fontweight="bold")

        # Reihe 1: Rf(p)
        ax1 = axes[0]
        rf_vals = df_g["Rf_p"].values.astype(float)
        colors  = ["#d62728" if (not np.isnan(v) and abs(v - 0.5) > 0.15) else "#2ca02c" for v in rf_vals]
        bars = ax1.bar(x, rf_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
        ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Fair (Rf=0.5)")
        for bar, val in zip(bars, rf_vals):
            if not np.isnan(val): ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Rf(p)  Frauenanteil")
        ax1.set_title("Gender Bias-Ratio Rf(p)  —  rot = starker Bias (|Rf−0.5| > 0.15)")
        ax1.legend(fontsize=8, loc="upper right")

        # Reihe 2: Chi²-Werte
        ax2 = axes[1]
        w = 0.35
        chi2_g = df_g["Chi2"].values.astype(float)
        chi2_e = df_e["Chi2"].values.astype(float)
        ax2.bar(x - w / 2, chi2_g, width=w, label="Gender",    color="#5b9bd5", edgecolor="black", linewidth=0.5)
        ax2.bar(x + w / 2, chi2_e, width=w, label="Ethnicity", color="#ed7d31", edgecolor="black", linewidth=0.5)
        ax2.axhline(CHI2_CRIT_GENDER,    color="#5b9bd5", linestyle="--", linewidth=1.2, alpha=0.8)
        ax2.axhline(CHI2_CRIT_ETHNICITY, color="#ed7d31", linestyle="--", linewidth=1.2, alpha=0.8)

        for i, (cg, ce) in enumerate(zip(chi2_g, chi2_e)):
            if not np.isnan(cg) and cg > CHI2_CRIT_GENDER: ax2.text(i - w / 2, cg + 0.3, "★", ha="center", fontsize=11, color="#5b9bd5")
            if not np.isnan(ce) and ce > CHI2_CRIT_ETHNICITY: ax2.text(i + w / 2, ce + 0.3, "★", ha="center", fontsize=11, color="#ed7d31")
        ax2.set_ylabel("Chi²-Wert")
        ax2.set_title("Chi²-Test  —  ★ = signifikant (p<0.05)")
        ax2.legend(fontsize=8, loc="upper right")

        # Reihe 3: Entropy
        ax3 = axes[2]
        ent_g = df_g["Entropy_norm"].values.astype(float)
        ent_e = df_e["Entropy_norm"].values.astype(float)
        ax3.bar(x - w / 2, ent_g, width=w, label="Gender",    color="#5b9bd5", edgecolor="black", linewidth=0.5)
        ax3.bar(x + w / 2, ent_e, width=w, label="Ethnicity", color="#ed7d31", edgecolor="black", linewidth=0.5)
        ax3.axhline(1.0, color="green", linestyle="--", linewidth=1.0, alpha=0.6, label="Max Diversität")
        for i, (eg, ee) in enumerate(zip(ent_g, ent_e)):
            if not np.isnan(eg): ax3.text(i - w / 2, eg + 0.02, f"{eg:.2f}", ha="center", fontsize=8)
            if not np.isnan(ee): ax3.text(i + w / 2, ee + 0.02, f"{ee:.2f}", ha="center", fontsize=8)
        ax3.set_ylim(0, 1.25)
        ax3.set_ylabel("Normalisierte Entropy")
        ax3.set_title("Diversitäts-Entropy  —  1.0 = maximal divers, 0.0 = eine Kategorie dominiert alles")
        ax3.legend(fontsize=8, loc="upper right")

        ax3.set_xticks(x)
        ax3.set_xticklabels(prompts, rotation=25, ha="right", fontsize=9)

        plt.tight_layout()
        fname = output_folder / f"PROMPT_OVERVIEW_COMBINED_{file_tag}_{model}.png"
        plt.savefig(fname, dpi=250, bbox_inches="tight")
        plt.close()

def plot_per_model_gender_only(df_stats: pd.DataFrame, output_folder: Path, label_title: str, file_tag: str):
    """NEU: Ein Plot exklusiv nur für GENDER (3 Reihen)"""
    models = df_stats["Modell"].dropna().unique()
    for model in sorted(models):
        df_m = df_stats[(df_stats["Modell"] == model) & (df_stats["Dimension"] == "Gender")].copy()
        if df_m.empty: continue
        prompts = sorted(df_m["Prompt"].dropna().unique())
        x = np.arange(len(prompts))
        df_g = df_m.set_index("Prompt").reindex(prompts)

        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(prompts) * 1.6), 13), sharex=True)
        fig.suptitle(f"[{label_title}]  Modell: {model.upper()}  —  GENDER Fokus", fontsize=13, fontweight="bold")

        # Reihe 1: Rf(p)
        ax1 = axes[0]
        rf_vals = df_g["Rf_p"].values.astype(float)
        colors  = ["#d62728" if (not np.isnan(v) and abs(v - 0.5) > 0.15) else "#2ca02c" for v in rf_vals]
        bars = ax1.bar(x, rf_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
        ax1.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Fair (Rf=0.5)")
        for bar, val in zip(bars, rf_vals):
            if not np.isnan(val): ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Rf(p)  Frauenanteil")
        ax1.set_title("Gender Bias-Ratio Rf(p)  —  rot = starker Bias (|Rf−0.5| > 0.15)")
        ax1.legend(fontsize=8, loc="upper right")

        # Reihe 2: Chi²-Werte
        ax2 = axes[1]
        chi2_g = df_g["Chi2"].values.astype(float)
        bars2 = ax2.bar(x, chi2_g, color="#5b9bd5", edgecolor="black", linewidth=0.5, width=0.55)
        ax2.axhline(CHI2_CRIT_GENDER, color="black", linestyle="--", linewidth=1.5, label=f"Signifikanzgrenze (χ²={CHI2_CRIT_GENDER})")
        for bar, cg in zip(bars2, chi2_g):
            if not np.isnan(cg) and cg > CHI2_CRIT_GENDER: 
                ax2.text(bar.get_x() + bar.get_width() / 2, cg + 0.3, "★", ha="center", fontsize=12, color="red")
        ax2.set_ylabel("Chi²-Wert")
        ax2.set_title("Chi²-Test (Gender)  —  ★ = signifikant (p<0.05)")
        ax2.legend(fontsize=8, loc="upper right")

        # Reihe 3: Entropy
        ax3 = axes[2]
        ent_g = df_g["Entropy_norm"].values.astype(float)
        bars3 = ax3.bar(x, ent_g, color="#5b9bd5", edgecolor="black", linewidth=0.5, width=0.55)
        ax3.axhline(1.0, color="green", linestyle="--", linewidth=1.5, alpha=0.6, label="Max Diversität (1.0)")
        for bar, eg in zip(bars3, ent_g):
            if not np.isnan(eg): ax3.text(bar.get_x() + bar.get_width() / 2, eg + 0.02, f"{eg:.2f}", ha="center", fontsize=9)
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Normalisierte Entropy")
        ax3.set_title("Diversitäts-Entropy (Gender)")
        ax3.legend(fontsize=8, loc="upper right")

        ax3.set_xticks(x)
        ax3.set_xticklabels(prompts, rotation=25, ha="right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_folder / f"PROMPT_OVERVIEW_GENDER_{file_tag}_{model}.png", dpi=250, bbox_inches="tight")
        plt.close()

def plot_per_model_ethnicity_only(df_stats: pd.DataFrame, output_folder: Path, label_title: str, file_tag: str):
    """NEU: Ein Plot exklusiv nur für ETHNICITY (3 Reihen, Dominanz statt Rf(p))"""
    models = df_stats["Modell"].dropna().unique()
    for model in sorted(models):
        df_m = df_stats[(df_stats["Modell"] == model) & (df_stats["Dimension"] == "Ethnicity")].copy()
        if df_m.empty: continue
        prompts = sorted(df_m["Prompt"].dropna().unique())
        x = np.arange(len(prompts))
        df_e = df_m.set_index("Prompt").reindex(prompts)

        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(prompts) * 1.6), 13), sharex=True)
        fig.suptitle(f"[{label_title}]  Modell: {model.upper()}  —  ETHNICITY Fokus", fontsize=13, fontweight="bold")

        # Reihe 1: Dominante Ethnie % (Ersatz für Rf(p))
        ax1 = axes[0]
        dom_pcts = df_e["Dominant_Pct"].values.astype(float)
        dom_cats = df_e["Dominant_Cat"].values
        colors = ["#d62728" if p > 70 else ("#ff7f0e" if p > 50 else "#2ca02c") for p in dom_pcts]
        bars = ax1.bar(x, dom_pcts, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
        ax1.axhline(100/len(ETHNICITY_CATS), color="red", linestyle="--", linewidth=1.5, label=f"Perfekte Fairheit ({100/len(ETHNICITY_CATS):.1f}%)")
        for bar, pct, cat in zip(bars, dom_pcts, dom_cats):
            if not np.isnan(pct): 
                ax1.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.0f}%\n{cat}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax1.set_ylim(0, 110)
        ax1.set_ylabel("Anteil der stärksten Ethnie (%)")
        ax1.set_title("Mode Collapse: Anteil der dominierenden Ethnie  (rot = extrem dominiert > 70%)")
        ax1.legend(fontsize=8, loc="upper right")

        # Reihe 2: Chi²-Werte
        ax2 = axes[1]
        chi2_e = df_e["Chi2"].values.astype(float)
        bars2 = ax2.bar(x, chi2_e, color="#ed7d31", edgecolor="black", linewidth=0.5, width=0.55)
        ax2.axhline(CHI2_CRIT_ETHNICITY, color="black", linestyle="--", linewidth=1.5, label=f"Signifikanzgrenze (χ²={CHI2_CRIT_ETHNICITY})")
        for bar, ce in zip(bars2, chi2_e):
            if not np.isnan(ce) and ce > CHI2_CRIT_ETHNICITY: 
                ax2.text(bar.get_x() + bar.get_width() / 2, ce + 0.5, "★", ha="center", fontsize=12, color="red")
        ax2.set_ylabel("Chi²-Wert")
        ax2.set_title("Chi²-Test (Ethnicity)  —  ★ = signifikant (p<0.05)")
        ax2.legend(fontsize=8, loc="upper right")

        # Reihe 3: Entropy
        ax3 = axes[2]
        ent_e = df_e["Entropy_norm"].values.astype(float)
        bars3 = ax3.bar(x, ent_e, color="#ed7d31", edgecolor="black", linewidth=0.5, width=0.55)
        ax3.axhline(1.0, color="green", linestyle="--", linewidth=1.5, alpha=0.6, label="Max Diversität (1.0)")
        for bar, ee in zip(bars3, ent_e):
            if not np.isnan(ee): ax3.text(bar.get_x() + bar.get_width() / 2, ee + 0.02, f"{ee:.2f}", ha="center", fontsize=9)
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Normalisierte Entropy")
        ax3.set_title("Diversitäts-Entropy (Ethnicity)")
        ax3.legend(fontsize=8, loc="upper right")

        ax3.set_xticks(x)
        ax3.set_xticklabels(prompts, rotation=25, ha="right", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_folder / f"PROMPT_OVERVIEW_ETHNICITY_{file_tag}_{model}.png", dpi=250, bbox_inches="tight")
        plt.close()


# =============================================================
# MASTER-EINSTIEGSPUNKT
# =============================================================
def generate_per_prompt_statistics(
    master_df: pd.DataFrame,
    output_folder: Path,
    dataset_label: str = "FAIR",
):
    print(f"\n[TEIL 4] Per-Prompt Statistiken [{dataset_label}] werden für alle VLMs berechnet...")

    if master_df.empty:
        print("  Leerer DataFrame — übersprungen.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    for vlm in VLM_LIST:
        vlm_safe_name = vlm.split('/')[-1].replace(':', '_')
        label_title   = f"{dataset_label} | VLM: {vlm_safe_name}"
        file_tag      = f"{dataset_label}_{vlm_safe_name}"
        
        print(f"\n  ➤ Verarbeite Statistik für: {vlm_safe_name}")

        df_stats = compute_per_prompt_metrics(master_df, vlm)

        if df_stats.empty:
            print(f"    Keine Daten für {vlm_safe_name} gefunden.")
            continue

        # CSV speichern
        csv_path = output_folder / f"PER_PROMPT_STATS_DETAIL_{file_tag}.csv"
        df_stats.to_csv(csv_path, index=False)
        print(f"    Detailtabelle: {csv_path.name}")

        # Plots generieren (Jetzt 4 Stück pro Modell!)
        plot_chi2_heatmaps(df_stats, output_folder, label_title, file_tag)
        plot_per_model_overview_combined(df_stats, output_folder, label_title, file_tag)
        plot_per_model_gender_only(df_stats, output_folder, label_title, file_tag)
        plot_per_model_ethnicity_only(df_stats, output_folder, label_title, file_tag)

    print(f"\n  Alle Per-Prompt Statistiken für [{dataset_label}] fertig -> {output_folder}")