import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CSV_PATH = "/home/mauli/repos/CAS13b_pipeline/per_subtype/spacers_conservation_filtered_optimized_predicted_subtypes.csv"


TOTAL_BY_TYPE = {
    "type1": 14906,
    "type2": 11975,
    "type3": 4897,
    "type4": 2868,
    "unclassified": 3411,
}

TYPE_ORDER = ["type1", "type2", "type3", "type4", "unclassified"]

# ----------------------------------------------------------------------
# LOAD AND PREPARE
# ----------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Check for required columns
required_cols = ["predicted_efficiency", "top_k_subtype", "type1", "type2", "type3", "type4", "unclassified"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Use a consistent spacer column name
if "optimized_spacer" in df.columns:
    df["spacer_seq"] = df["optimized_spacer"]
elif "spacer" in df.columns:
    df["spacer_seq"] = df["spacer"]
else:
    raise ValueError("Expected 'optimized_spacer' or 'spacer' column in CSV.")

# Compute per-type conservation percentages for each spacer
for t in TYPE_ORDER:
    df[f"{t}_conservation_pct"] = 100.0 * df[t] / TOTAL_BY_TYPE[t]

print(f"\nTotal rows in CSV: {len(df)}")
print(f"Unique top_k_subtype values: {df['top_k_subtype'].unique()}")
print(f"Distribution of top_k_subtype:")
print(df['top_k_subtype'].value_counts())

# ----------------------------------------------------------------------
# PER-TYPE STATS AND BEST CANDIDATES (FILTERED BY top_k_subtype)
# ----------------------------------------------------------------------

def summarize_type_filtered(df, type_col):
    """
    Return per-type summary and best candidates.
    Filter to rows where top_k_subtype matches type_col.
    """
    # Filter to rows belonging to this type
    df_type = df[df['top_k_subtype'] == type_col].copy()
    
    if df_type.empty:
        print(f"\nWARNING: No rows with top_k_subtype == '{type_col}'")
        return None
    
    N_total = TOTAL_BY_TYPE[type_col]
    
    # Most conserved candidate (highest type-specific conservation within this type's candidates)
    most_conserved = df_type.loc[df_type[f"{type_col}_conservation_pct"].idxmax()]
    
    # Best efficiency candidate within this type
    best_eff = df_type.loc[df_type["predicted_efficiency"].idxmax()]
    
    # Per-type-level stats
    summary = {
        "num_candidates_this_type": len(df_type),
        "mean_eff": df_type["predicted_efficiency"].mean(),
        "median_eff": df_type["predicted_efficiency"].median(),
        "std_eff": df_type["predicted_efficiency"].std(),
        "mean_conservation_pct": df_type[f"{type_col}_conservation_pct"].mean(),
        "max_conservation_pct": df_type[f"{type_col}_conservation_pct"].max(),
        "min_conservation_pct": df_type[f"{type_col}_conservation_pct"].min(),
    }
    
    return {
        "summary": summary,
        "most_conserved": most_conserved,
        "best_eff": best_eff,
        "df_type": df_type,
    }

type_results = {}
print("\n" + "="*90)
print("PER-SEROTYPE STATISTICS AND BEST CANDIDATES (FILTERED BY top_k_subtype)")
print("="*90)

for t in TYPE_ORDER:
    res = summarize_type_filtered(df, t)
    if res is None:
        continue
    type_results[t] = res
    s = res["summary"]
    mc = res["most_conserved"]
    be = res["best_eff"]
    
    print(f"\n{t.upper()}")
    print("-"*90)
    print(f"Total genomes of this type               : {TOTAL_BY_TYPE[t]:,}")
    print(f"Candidates in top_k list for this type   : {s['num_candidates_this_type']} "
          f"({100*s['num_candidates_this_type']/len(df):.1f}% of all candidates)")
    print(f"\nEfficiency Statistics (for this type's top_k candidates):")
    print(f"  Mean predicted efficiency              : {s['mean_eff']:.3f}")
    print(f"  Median predicted efficiency            : {s['median_eff']:.3f}")
    print(f"  Std Dev                                : {s['std_eff']:.3f}")
    print(f"\nPer-type Conservation (for this type's top_k candidates):")
    print(f"  Mean {t} conservation                  : {s['mean_conservation_pct']:.2f}%")
    print(f"  Max  {t} conservation                  : {s['max_conservation_pct']:.2f}%")
    print(f"  Min  {t} conservation                  : {s['min_conservation_pct']:.2f}%")
    
    print(f"\n  → MOST CONSERVED CANDIDATE (within {t} top_k list):")
    print(f"    Spacer: {mc['spacer_seq']}")
    print(f"    {t} count / total                     : {int(mc[t])} / {TOTAL_BY_TYPE[t]} "
          f"({mc[f'{t}_conservation_pct']:.2f}%)")
    print(f"    Predicted efficiency                  : {mc['predicted_efficiency']:.3f}")
    
    print(f"\n  → BEST-EFFICIENCY CANDIDATE (within {t} top_k list):")
    print(f"    Spacer: {be['spacer_seq']}")
    print(f"    Predicted efficiency                  : {be['predicted_efficiency']:.3f}")
    print(f"    {t} count / total                     : {int(be[t])} / {TOTAL_BY_TYPE[t]} "
          f"({be[f'{t}_conservation_pct']:.2f}%)")

# ----------------------------------------------------------------------
# FIGURE 1: Efficiency vs per-type conservation (scatter, filtered by top_k)
# ----------------------------------------------------------------------

def plot_eff_vs_conservation_filtered(type_results, output="fig_eff_vs_conservation_filtered.png"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    type_order = ["type1", "type2", "type3", "type4"]
    
    for i, t in enumerate(type_order):
        ax = axes[i]
        res = type_results.get(t)
        if res is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data for {t}", ha='center', va='center')
            continue
        
        dft = res["df_type"]
        sc = ax.scatter(dft[f"{t}_conservation_pct"], dft["predicted_efficiency"],
                        alpha=0.6, s=50, edgecolor="black", linewidth=0.5, c=dft["predicted_efficiency"],
                        cmap="viridis")
        
        # Mark best candidates
        mc = res['most_conserved']
        be = res['best_eff']
        ax.scatter([mc[f'{t}_conservation_pct']], [mc['predicted_efficiency']], 
                  marker='*', s=500, color='red', edgecolor='darkred', linewidth=1.5, 
                  label='Most conserved', zorder=5)
        ax.scatter([be[f'{t}_conservation_pct']], [be['predicted_efficiency']], 
                  marker='D', s=150, color='lime', edgecolor='darkgreen', linewidth=1.5, 
                  label='Best efficiency', zorder=5)
        
        ax.set_title(f"{t.upper()} (n={len(dft)})", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{t} conservation (%)", fontsize=10)
        ax.set_ylabel("Predicted efficiency", fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.3, linewidth=1)
        ax.legend(fontsize=8, loc='best')
    
    # Turn off last empty axis
    # axes[-1].axis("off")
    
    fig.suptitle("Efficiency vs Per-Type Conservation (Filtered by top_k_subtype)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {output}")

plot_eff_vs_conservation_filtered(type_results)

# ----------------------------------------------------------------------
# FIGURE 2: Bar plot of top candidates (most conserved vs best efficiency)
# ----------------------------------------------------------------------

def plot_best_candidates_bar_filtered(type_results, output="fig_best_candidates_bar_filtered.png"):
    rows = []
    for t, res in type_results.items():
        mc = res["most_conserved"]
        be = res["best_eff"]
        rows.append({
            "type": t,
            "kind": "Most conserved",
            "spacer": mc["spacer_seq"],
            "eff": mc["predicted_efficiency"],
            "conservation": mc[f"{t}_conservation_pct"],
        })
        rows.append({
            "type": t,
            "kind": "Best efficiency",
            "spacer": be["spacer_seq"],
            "eff": be["predicted_efficiency"],
            "conservation": be[f"{t}_conservation_pct"],
        })
    
    plot_df = pd.DataFrame(rows)
    plot_df["Type"] = plot_df["type"].str.replace("type", "DENV-").str.replace("unclassified", "Unclassified")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Efficiency
    sns.barplot(data=plot_df, x="Type", y="eff", hue="kind", ax=axes[0],
                palette=["#1f77b4", "#ff7f0e"], width=0.7)
    axes[0].set_ylabel("Predicted efficiency", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Dengue serotype", fontsize=11, fontweight="bold")
    axes[0].set_title("Top candidates per type: Predicted Efficiency", fontsize=12, fontweight="bold")
    axes[0].axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, label="High efficacy (0.8)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(title="Candidate Type", fontsize=10)
    axes[0].set_ylim([0, 1.0])
    
    # Conservation
    sns.barplot(data=plot_df, x="Type", y="conservation", hue="kind", ax=axes[1],
                palette=["#1f77b4", "#ff7f0e"], width=0.7)
    axes[1].set_ylabel("Per-type conservation (%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Dengue serotype", fontsize=11, fontweight="bold")
    axes[1].set_title("Top candidates per type: Conservation Coverage", fontsize=12, fontweight="bold")
    axes[1].axhline(80, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, label="High conservation (80%)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(title="Candidate Type", fontsize=10)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output}")

plot_best_candidates_bar_filtered(type_results)

# ----------------------------------------------------------------------
# FIGURE 3: Distribution of per-type conservation (violin plots, filtered)
# ----------------------------------------------------------------------

def plot_conservation_distributions_filtered(type_results, output="fig_conservation_distributions_filtered.png"):
    melted = []
    for t, res in type_results.items():
        dft = res["df_type"]
        melted.append(
            dft[[f"{t}_conservation_pct"]]
            .rename(columns={f"{t}_conservation_pct": "conservation_pct"})
            .assign(type=t.replace("type", "DENV-").replace("unclassified", "Unclassified"))
        )
    melted_df = pd.concat(melted, ignore_index=True)
    
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=melted_df, x="type", y="conservation_pct", inner="quartile",
                   order=[t.replace("type", "DENV-").replace("unclassified", "Unclassified") for t in TYPE_ORDER if t in type_results],
                   palette="Set2")
    plt.xlabel("Dengue serotype", fontsize=11, fontweight="bold")
    plt.ylabel("Per-type conservation (%)", fontsize=11, fontweight="bold")
    plt.title("Distribution of per-type conservation among top_k candidate crRNAs", 
             fontsize=13, fontweight="bold")
    plt.axhline(80, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output}")

plot_conservation_distributions_filtered(type_results)

# ----------------------------------------------------------------------
# FIGURE 4: Efficiency distribution by serotype (histograms)
# ----------------------------------------------------------------------

def plot_efficiency_distributions_filtered(type_results, output="fig_efficiency_distributions_filtered.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, t in enumerate(TYPE_ORDER):
        ax = axes[i]
        res = type_results.get(t)
        if res is None:
            ax.axis("off")
            continue
        
        dft = res["df_type"]
        ax.hist(dft["predicted_efficiency"], bins=20, color="steelblue", 
               edgecolor="black", alpha=0.7)
        
        mean_eff = dft["predicted_efficiency"].mean()
        median_eff = dft["predicted_efficiency"].median()
        
        ax.axvline(mean_eff, color="red", linestyle="--", linewidth=2, 
                  label=f"Mean: {mean_eff:.3f}")
        ax.axvline(median_eff, color="orange", linestyle="--", linewidth=2, 
                  label=f"Median: {median_eff:.3f}")
        ax.axvline(0.8, color="green", linestyle=":", linewidth=2, alpha=0.7,
                  label="High efficacy (0.8)")
        
        ax.set_xlabel("Predicted efficiency", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax.set_title(f"{t.upper()} (n={len(dft)})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    
    axes[-1].axis("off")
    fig.suptitle("Distribution of Predicted Efficiency by Serotype (top_k filtered)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output}")

plot_efficiency_distributions_filtered(type_results)

# ----------------------------------------------------------------------
# SUMMARY FOR PAPER
# ----------------------------------------------------------------------

print("\n" + "="*90)
print("SUMMARY OF FINDINGS FOR PAPER RESULTS SECTION")
print("="*90)

summary_text = ""

for t in TYPE_ORDER:
    res = type_results.get(t)
    if not res: 
        continue
    
    mc = res['most_conserved']
    be = res['best_eff']
    s = res['summary']
    t_display = t.replace("type", "DENV-").replace("unclassified", "Unclassified")
    
    summary_text += f"\n{t_display.upper()}:\n"
    summary_text += f"  • Identified {s['num_candidates_this_type']} top_k candidates for {t_display}.\n"
    summary_text += f"  • Mean predicted efficiency: {s['mean_eff']:.3f} (±{s['std_eff']:.3f})\n"
    summary_text += f"  • Highest-conserved candidate: {mc['spacer_seq']}\n"
    summary_text += f"    - Coverage: {int(mc[t])} / {TOTAL_BY_TYPE[t]} genomes ({mc[f'{t}_conservation_pct']:.2f}%)\n"
    summary_text += f"    - Predicted efficiency: {mc['predicted_efficiency']:.3f}\n"
    summary_text += f"  • Best-efficiency candidate: {be['spacer_seq']}\n"
    summary_text += f"    - Predicted efficiency: {be['predicted_efficiency']:.3f}\n"
    summary_text += f"    - Coverage: {int(be[t])} / {TOTAL_BY_TYPE[t]} genomes ({be[f'{t}_conservation_pct']:.2f}%)\n"

print(summary_text)
