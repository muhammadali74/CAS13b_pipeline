import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_optimization_results(input_csv, output_png='optimization_analysis.png'):
    """
    Analyze Cas13b spacer optimization results and generate visualization.
    
    Parameters
    ----------
    input_csv : str
        Path to CSV with optimization results (columns: original_score, optimized_score, score_improvement, etc.)
    output_png : str
        Path to save the output figure
    
    Returns
    -------
    dict : Statistics including average and maximum score improvement
    """
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Calculate statistics
    num_spacers = len(df)
    num_improved = (df['score_improvement'] > 0).sum()
    pct_improved = 100 * num_improved / num_spacers
    
    avg_improvement = df['score_improvement'].mean()
    max_improvement = df['score_improvement'].max()
    min_improvement = df['score_improvement'].min()
    std_improvement = df['score_improvement'].std()
    
    # Count mutations
    mutation_counts = df['num_mutations'].value_counts().sort_index()
    
    # Analyze which positions are most commonly mutated
    mutation_positions = []
    for mutations_str in df['mutations'].dropna():
        if isinstance(mutations_str, str) and mutations_str.strip() != '':
            muts = mutations_str.split(',')
            for mut in muts:
                # Extract position from mutation string (e.g., "C1G" -> 1)
                pos = ''.join(filter(str.isdigit, mut))
                if pos:
                    mutation_positions.append(int(pos))
    
    position_mutation_freq = {}
    for pos in mutation_positions:
        position_mutation_freq[pos] = position_mutation_freq.get(pos, 0) + 1
    
    # Print summary statistics
    print("="*70)
    print("CAS13B SPACER OPTIMIZATION ANALYSIS")
    print("="*70)
    print(f"\nTotal spacers analyzed: {num_spacers}")
    print(f"Spacers improved: {num_improved} ({pct_improved:.1f}%)")
    print(f"Already optimal spacers: {num_spacers - num_improved} ({100 - pct_improved:.1f}%)")
    print(f"\nScore Improvement Statistics:")
    print(f"  Average improvement: +{avg_improvement:.2f} points")
    print(f"  Maximum improvement: +{max_improvement:.0f} points")
    print(f"  Minimum improvement: {min_improvement:.0f} points")
    print(f"  Standard deviation: {std_improvement:.2f}")
    print(f"\nMutation Distribution:")
    for num_mut, count in mutation_counts.items():
        print(f"  Spacers with {int(num_mut)} mutation(s): {count} ({100*count/num_spacers:.1f}%)")
    
    print(f"\nMost frequently mutated positions:")
    top_positions = sorted(position_mutation_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for pos, freq in top_positions:
        print(f"  Position {pos}: {freq} mutations")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Score improvement histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df['score_improvement'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(avg_improvement, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_improvement:.2f}')
    ax1.axvline(0, color='orange', linestyle='--', linewidth=2, label='No improvement')
    ax1.set_xlabel('Score Improvement (points)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Score Improvements', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Original vs Optimized scatter plot
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(df['original_score'], df['optimized_score'], 
                         c=df['score_improvement'], cmap='viridis', alpha=0.6, s=30)
    # Diagonal line (no change)
    min_score = min(df['original_score'].min(), df['optimized_score'].min())
    max_score = max(df['original_score'].max(), df['optimized_score'].max())
    ax2.plot([min_score, max_score], [min_score, max_score], 'r--', linewidth=2, label='No change')
    ax2.set_xlabel('Original Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Optimized Score', fontsize=11, fontweight='bold')
    ax2.set_title('Original vs Optimized Scores', fontsize=12, fontweight='bold')
    ax2.legend()
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Score Improvement', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Mutation distribution
    ax3 = plt.subplot(2, 3, 3)
    mutation_counts.plot(kind='bar', ax=ax3, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Number of Mutations', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Mutation Count Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(mutation_counts.index, rotation=0)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Score improvement by mutation count
    ax4 = plt.subplot(2, 3, 4)
    mutation_improvement = df.groupby('num_mutations')['score_improvement'].agg(['mean', 'std'])
    x_pos = range(len(mutation_improvement))
    ax4.bar(x_pos, mutation_improvement['mean'], yerr=mutation_improvement['std'],
            color='lightgreen', edgecolor='black', alpha=0.7, capsize=5)
    ax4.set_xlabel('Number of Mutations', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Score Improvement (±SD)', fontsize=11, fontweight='bold')
    ax4.set_title('Score Improvement by Mutation Count', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(mutation_improvement.index)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Most frequently mutated positions
    ax5 = plt.subplot(2, 3, 5)
    if position_mutation_freq:
        top_n = 15
        top_positions_dict = dict(sorted(position_mutation_freq.items(), 
                                         key=lambda x: x[1], reverse=True)[:top_n])
        positions = list(top_positions_dict.keys())
        freqs = list(top_positions_dict.values())
        colors = ['red' if p in [1, 2] else 'orange' if p in [11, 12, 15, 16, 17] else 'steelblue' 
                  for p in positions]
        ax5.barh(range(len(positions)), freqs, color=colors, edgecolor='black', alpha=0.7)
        ax5.set_yticks(range(len(positions)))
        ax5.set_yticklabels(positions)
        ax5.set_xlabel('Frequency of Mutation', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Position', fontsize=11, fontweight='bold')
        ax5.set_title('Top 15 Most Frequently Mutated Positions', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='5\' end (Pos 1-2)'),
            Patch(facecolor='orange', edgecolor='black', label='Central region (Pos 11,12,15-17)'),
            Patch(facecolor='steelblue', edgecolor='black', label='Other positions')
        ]
        ax5.legend(handles=legend_elements, loc='lower right', fontsize=9)
        ax5.grid(axis='x', alpha=0.3)
    
    # 6. Conservation vs Score Improvement
    ax6 = plt.subplot(2, 3, 6)
    scatter2 = ax6.scatter(df['conservation_pct'], df['score_improvement'], 
                          c=df['optimized_score'], cmap='plasma', alpha=0.6, s=30)
    ax6.set_xlabel('Conservation (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Score Improvement (points)', fontsize=11, fontweight='bold')
    ax6.set_title('Conservation vs Score Improvement', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax6)
    cbar2.set_label('Optimized Score', fontweight='bold')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_png}")
    plt.close()
    
    # Return statistics dictionary
    stats = {
        'total_spacers': num_spacers,
        'spacers_improved': num_improved,
        'pct_improved': pct_improved,
        'avg_improvement': avg_improvement,
        'max_improvement': max_improvement,
        'min_improvement': min_improvement,
        'std_improvement': std_improvement,
        'mutation_counts': mutation_counts.to_dict(),
        'position_mutation_freq': position_mutation_freq
    }
    
    return stats


def generate_summary_table(stats):
    """Generate a summary table of optimization results."""
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Metric':<40} {'Value':<20}")
    print("-"*70)
    print(f"{'Total spacers analyzed':<40} {stats['total_spacers']:<20}")
    print(f"{'Spacers improved':<40} {stats['spacers_improved']} ({stats['pct_improved']:.1f}%)")
    print(f"{'Already optimal':<40} {stats['total_spacers'] - stats['spacers_improved']:<20}")
    print(f"{'Average score improvement':<40} {stats['avg_improvement']:.2f} points")
    print(f"{'Maximum score improvement':<40} {stats['max_improvement']:.0f} points")
    print(f"{'Minimum score improvement':<40} {stats['min_improvement']:.0f} points")
    print(f"{'Standard deviation':<40} {stats['std_improvement']:.2f}")
    print("="*70 + "\n")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Run analysis
    stats = analyze_optimization_results(
        input_csv='/home/mauli/repos/CAS13b_pipeline/per_subtype/spacers_conservation_filtered_optimized_subtypes.csv',
        output_png='optimization_analysis.png'
    )
    
    # Print summary
    generate_summary_table(stats)