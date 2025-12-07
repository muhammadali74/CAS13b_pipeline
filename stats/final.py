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

# make sure expected columns exist
required_cols = ["predicted_efficiency", "type1", "type2", "type3", "type4", "unclassified"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# use a consistent spacer column name
if "optimized_spacer" in df.columns:
    df["spacer_seq"] = df["optimized_spacer"]
elif "spacer" in df.columns:
    df["spacer_seq"] = df["spacer"]
else:
    raise ValueError("Expected 'optimized_spacer' or 'spacer' column in CSV.")

# compute per-type conservation percentages for each spacer
for t in TYPE_ORDER:
    df[f"{t}_conservation_pct"] = 100.0 * df[t] / TOTAL_BY_TYPE[t]

# ----------------------------------------------------------------------
# PER-TYPE STATS AND BEST CANDIDATES
# ----------------------------------------------------------------------

def summarize_type(df, type_col):
    """Return per-type summary and best candidates."""
    N_total = TOTAL_BY_TYPE[type_col]
    
    # Only candidates that appear at least once in this type
    df_type = df[df[type_col] > 0].copy()
    
    if df_type.empty:
        return None
    
    # Most conserved candidate (highest type-specific conservation)
    most_conserved = df_type.loc[df_type[f"{type_col}_conservation_pct"].idxmax()]
    
    # Best efficiency candidate within this type
    best_eff = df_type.loc[df_type["predicted_efficiency"].idxmax()]
    
    # type-level stats
    summary = {
        "num_candidates_this_type": len(df_type),
        "mean_eff": df_type["predicted_efficiency"].mean(),
        "median_eff": df_type["predicted_efficiency"].median(),
        "mean_conservation_pct": df_type[f"{type_col}_conservation_pct"].mean(),
        "max_conservation_pct": df_type[f"{type_col}_conservation_pct"].max(),
    }
    
    return {
        "summary": summary,
        "most_conserved": most_conserved,
        "best_eff": best_eff,
        "df_type": df_type,
    }

type_results = {}
print("\n" + "="*80)
print("PER-SEROTYPE STATISTICS AND BEST CANDIDATES")
print("="*80)

for t in TYPE_ORDER:
    res = summarize_type(df, t)
    if res is None:
        print(f"\n{t.upper()}: No candidates with non-zero count.")
        continue
    type_results[t] = res
    s = res["summary"]
    mc = res["most_conserved"]
    be = res["best_eff"]
    
    print(f"\n{t.upper()}")
    print("-"*80)
    print(f"Total genomes of this type               : {TOTAL_BY_TYPE[t]}")
    print(f"Candidates occurring in this type        : {s['num_candidates_this_type']} "
          f"({100*s['num_candidates_this_type']/len(df):.1f}% of all candidates)")
    print(f"Mean predicted efficiency                : {s['mean_eff']:.3f}")
    print(f"Median predicted efficiency              : {s['median_eff']:.3f}")
    print(f"Mean {t} conservation (per-candidate)    : {s['mean_conservation_pct']:.2f}%")
    print(f"Max  {t} conservation                    : {s['max_conservation_pct']:.2f}%")
    
    print("\n  Most conserved candidate:")
    print(f"    Spacer: {mc['spacer_seq']}")
    print(f"    {t} count / total                     : {mc[t]} / {TOTAL_BY_TYPE[t]} "
          f"({mc[f'{t}_conservation_pct']:.2f}%)")
    print(f"    Predicted efficiency                  : {mc['predicted_efficiency']:.3f}")
    
    print("\n  Best-efficiency candidate in this type:")
    print(f"    Spacer: {be['spacer_seq']}")
    print(f"    Predicted efficiency                  : {be['predicted_efficiency']:.3f}")
    print(f"    {t} count / total                     : {be[t]} / {TOTAL_BY_TYPE[t]} "
          f"({be[f'{t}_conservation_pct']:.2f}%)")

# ----------------------------------------------------------------------
# FIGURE 1: Efficiency vs per-type conservation (scatter) 
# ----------------------------------------------------------------------

def plot_eff_vs_conservation(df, type_results, output="fig_eff_vs_conservation.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, t in enumerate(TYPE_ORDER):
        ax = axes[i]
        res = type_results.get(t)
        if res is None:
            ax.axis("off")
            continue
        
        dft = res["df_type"]
        sc = ax.scatter(dft[f"{t}_conservation_pct"], dft["predicted_efficiency"],
                        alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(t.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{t} conservation (%)", fontsize=11)
        ax.set_ylabel("Predicted efficiency", fontsize=11)
        ax.grid(alpha=0.3)
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5)
    
    # Turn off last empty axis (if any)
    if len(TYPE_ORDER) < len(axes):
        axes[-1].axis("off")
    
    fig.suptitle("Efficiency vs Per-Type Conservation for Dengue Cas13b crRNA Candidates",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output}")

plot_eff_vs_conservation(df, type_results)

# ----------------------------------------------------------------------
# FIGURE 2: Barplot of top-most conserved and best-efficiency candidates per type
# ----------------------------------------------------------------------

def plot_best_candidates_bar(type_results, output="fig_best_candidates_bar.png"):
    rows = []
    for t, res in type_results.items():
        mc = res["most_conserved"]
        be = res["best_eff"]
        rows.append({
            "type": t,
            "kind": "most_conserved",
            "spacer": mc["spacer_seq"],
            "eff": mc["predicted_efficiency"],
            "conservation": mc[f"{t}_conservation_pct"],
        })
        rows.append({
            "type": t,
            "kind": "best_efficiency",
            "spacer": be["spacer_seq"],
            "eff": be["predicted_efficiency"],
            "conservation": be[f"{t}_conservation_pct"],
        })
    
    plot_df = pd.DataFrame(rows)
    # For readability
    plot_df["Type"] = plot_df["type"].str.replace("type", "DENV-")
    plot_df["Candidate"] = plot_df["kind"].map({
        "most_conserved": "Most conserved",
        "best_efficiency": "Best efficiency"
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Efficiency
    sns.barplot(data=plot_df, x="Type", y="eff", hue="Candidate", ax=axes[0],
                palette=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylabel("Predicted efficiency", fontsize=11)
    axes[0].set_title("Best candidates per type: efficiency", fontsize=12, fontweight="bold")
    axes[0].axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Conservation
    sns.barplot(data=plot_df, x="Type", y="conservation", hue="Candidate", ax=axes[1],
                palette=["#1f77b4", "#ff7f0e"])
    axes[1].set_ylabel("Per-type conservation (%)", fontsize=11)
    axes[1].set_title("Best candidates per type: conservation", fontsize=12, fontweight="bold")
    axes[1].axhline(80, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")

plot_best_candidates_bar(type_results)

# ----------------------------------------------------------------------
# FIGURE 3: Distribution of per-type conservation across candidates
# ----------------------------------------------------------------------

def plot_conservation_distributions(df, type_order, output="fig_conservation_distributions.png"):
    plt.figure(figsize=(12, 8))
    melted = []
    for t in type_order:
        melted.append(
            df[[f"{t}_conservation_pct"]]
            .rename(columns={f"{t}_conservation_pct": "conservation_pct"})
            .assign(type=t)
        )
    melted_df = pd.concat(melted, ignore_index=True)
    
    melted_df = melted_df[melted_df["conservation_pct"] > 0]  # only spacers present in that type
    
    sns.violinplot(data=melted_df, x="type", y="conservation_pct", inner="quartile",
                   order=type_order, palette="Set2")
    plt.xlabel("Dengue type", fontsize=11)
    plt.ylabel("Per-type conservation (%)", fontsize=11)
    plt.title("Distribution of per-type conservation among candidate crRNAs", fontsize=13, fontweight="bold")
    plt.axhline(80, color="gray", linestyle="--", alpha=0.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")

plot_conservation_distributions(df, TYPE_ORDER)

