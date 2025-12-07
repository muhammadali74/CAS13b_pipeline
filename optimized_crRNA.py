import pandas as pd
from itertools import combinations, product
import numpy as np

def score_cas13b_spacer(spacer_seq):
    """
    Score a 30-nt crRNA spacer sequence using the algorithm from Hu et al. 2022.
    
    Returns
    -------
    dict with score, prediction, and details
    """
    spacer = spacer_seq.upper().replace('U', 'T')
    
    if len(spacer) != 30:
        raise ValueError(f"Spacer must be 30 nucleotides. Got {len(spacer)}")
    
    score = 0
    details = []
    
    # Rule 1: G at position 1 or 2 (+20 each)
    if spacer[0] == 'G':
        score += 20
        details.append(('pos_1_G', +20))
    if spacer[1] == 'G':
        score += 20
        details.append(('pos_2_G', +20))
    
    # Rule 2: C at positions 1-4 (-20 each)
    for i in range(4):
        if spacer[i] == 'C':
            score -= 20
            details.append((f'pos_{i+1}_C', -20))
    
    # Rule 3: C at positions 11, 12, 15, 16, 17 (-5 each)
    penalty_positions = [10, 11, 14, 15, 16]  # 0-indexed
    for pos in penalty_positions:
        if spacer[pos] == 'C':
            score -= 5
            details.append((f'pos_{pos+1}_C', -5))
    
    prediction = 'Potent' if score >= 0 else 'Ineffective'
    
    return {
        'spacer': spacer,
        'score': score,
        'prediction': prediction,
        'details': details
    }


def identify_optimal_mutations(spacer):
    """
    Identify positions where mutations would improve Cas13b score.
    
    Returns list of (position, current_nt, optimal_nt, score_gain)
    """
    mutations = []
    
    # Position 1: if not G, changing to G gives +20
    if spacer[0] != 'G':
        gain = 20
        if spacer[0] == 'C':
            gain += 20  # Also removes -20 penalty
        mutations.append((0, spacer[0], 'G', gain))
    
    # Position 2: if not G, changing to G gives +20
    if spacer[1] != 'G':
        gain = 20
        if spacer[1] == 'C':
            gain += 20  # Also removes -20 penalty
        mutations.append((1, spacer[1], 'G', gain))
    
    # Positions 3-4: if C, change to anything else gains +20
    for pos in [2, 3]:
        if spacer[pos] == 'C':
            # Change C to G (most beneficial - creates potential GG motif)
            mutations.append((pos, 'C', 'G', 20))
    
    # Positions 11, 12, 15, 16, 17: if C, change to anything else gains +5
    penalty_positions = [10, 11, 14, 15, 16]
    for pos in penalty_positions:
        if spacer[pos] == 'C':
            # Change C to G (conservative choice)
            mutations.append((pos, 'C', 'G', 5))
    
    # Sort by score gain (descending)
    mutations.sort(key=lambda x: x[3], reverse=True)
    
    return mutations


def generate_optimized_crRNAs(original_spacer, max_mismatches=3):
    """
    Generate optimized crRNAs with up to max_mismatches beneficial mutations.
    
    Returns list of optimized crRNA variants with their scores.
    """
    spacer = original_spacer.upper().replace('U', 'T')
    
    # Get original score
    original_result = score_cas13b_spacer(spacer)
    original_score = original_result['score']
    
    # Find all beneficial mutations
    possible_mutations = identify_optimal_mutations(spacer)
    
    if not possible_mutations:
        # No beneficial mutations possible
        return [{
            'original_spacer': original_spacer,
            'optimized_spacer': original_spacer,
            'original_score': original_score,
            'optimized_score': original_score,
            'num_mutations': 0,
            'mutations': '',
            'score_improvement': 0,
            'prediction': original_result['prediction']
        }]
    
    # Generate all combinations of mutations (up to max_mismatches)
    best_variants = []
    max_score = original_score
    
    for num_muts in range(1, min(max_mismatches, len(possible_mutations)) + 1):
        for mutation_combo in combinations(possible_mutations, num_muts):
            # Apply mutations
            modified_spacer = list(spacer)
            mutation_descriptions = []
            
            for pos, old_nt, new_nt, gain in mutation_combo:
                modified_spacer[pos] = new_nt
                mutation_descriptions.append(f"{old_nt}{pos+1}{new_nt}")
            
            modified_spacer_str = ''.join(modified_spacer)
            
            # Score the modified spacer
            modified_result = score_cas13b_spacer(modified_spacer_str)
            modified_score = modified_result['score']
            
            # Track if this is a best score
            if modified_score > max_score:
                max_score = modified_score
                best_variants = []  # Clear previous best
            
            if modified_score == max_score:
                best_variants.append({
                    'original_spacer': original_spacer,
                    'optimized_spacer': modified_spacer_str,
                    'original_score': original_score,
                    'optimized_score': modified_score,
                    'num_mutations': num_muts,
                    'mutations': ','.join(mutation_descriptions),
                    'score_improvement': modified_score - original_score,
                    'prediction': modified_result['prediction']
                })
    
    # If no improvement found, return original
    if not best_variants:
        return [{
            'original_spacer': original_spacer,
            'optimized_spacer': original_spacer,
            'original_score': original_score,
            'optimized_score': original_score,
            'num_mutations': 0,
            'mutations': '',
            'score_improvement': 0,
            'prediction': original_result['prediction']
        }]
    
    return best_variants


def optimize_spacers_csv(input_csv, output_csv, max_mismatches=3):
    """
    Read CSV with spacers, generate optimized crRNAs with up to max_mismatches,
    and save results.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV with 'spacer' column
    output_csv : str
        Path to save optimized results
    max_mismatches : int
        Maximum number of mutations allowed (default: 3)
    """
    df = pd.read_csv(input_csv)
    
    if 'spacer' not in df.columns:
        raise ValueError("CSV must have 'spacer' column")
    
    print(f"Optimizing {len(df)} spacers (max {max_mismatches} mismatches)...")
    print("This may take a few minutes...\n")
    
    all_results = []
    
    for idx, row in df.iterrows():
        spacer = row['spacer']
        
        # Generate optimized variants
        optimized_variants = generate_optimized_crRNAs(spacer, max_mismatches)
        
        # Add original row data to each variant
        for variant in optimized_variants:
            result = row.to_dict()
            result.update(variant)
            all_results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} spacers...")
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    priority_cols = [
        'spacer', 'original_spacer', 'optimized_spacer',
        'original_score', 'optimized_score', 'score_improvement',
        'num_mutations', 'mutations', 'prediction'
    ]
    
    # Add priority columns first, then remaining columns
    other_cols = [col for col in df_results.columns if col not in priority_cols]
    final_cols = [col for col in priority_cols if col in df_results.columns] + other_cols
    df_results = df_results[final_cols]
    
    # Sort by optimized score (descending)
    df_results = df_results.sort_values('optimized_score', ascending=False)
    
    # Save
    df_results.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total input spacers: {len(df)}")
    print(f"Total output variants: {len(df_results)}")
    print(f"  (some spacers have multiple equally-optimal variants)")
    
    improved = df_results[df_results['score_improvement'] > 0]
    print(f"\nSpacers improved: {len(improved)} ({100*len(improved)/len(df_results):.1f}%)")
    print(f"Average improvement: {improved['score_improvement'].mean():.1f} points")
    print(f"Max improvement: {df_results['score_improvement'].max():.0f} points")
    
    print(f"\nOriginal score distribution:")
    print(f"  Mean: {df_results['original_score'].mean():.1f}")
    print(f"  Range: {df_results['original_score'].min():.0f} to {df_results['original_score'].max():.0f}")
    
    print(f"\nOptimized score distribution:")
    print(f"  Mean: {df_results['optimized_score'].mean():.1f}")
    print(f"  Range: {df_results['optimized_score'].min():.0f} to {df_results['optimized_score'].max():.0f}")
    
    print(f"\nMutation distribution:")
    print(df_results['num_mutations'].value_counts().sort_index())
    
    print(f"\nTop 10 optimized crRNAs:")
    print(df_results[['optimized_spacer', 'optimized_score', 'num_mutations', 'mutations']].head(10).to_string(index=False))
    
    print(f"\nResults saved to: {output_csv}")
    print(f"{'='*70}\n")
    
    return df_results


def compare_original_vs_optimized(df_results):
    """
    Generate comparison report between original and optimized spacers.
    """
    import matplotlib.pyplot as plt
    
    # Score improvement histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df_results['score_improvement'], bins=20, edgecolor='black')
    plt.xlabel('Score Improvement')
    plt.ylabel('Count')
    plt.title('Distribution of Score Improvements')
    plt.axvline(0, color='red', linestyle='--', label='No improvement')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(df_results['original_score'], df_results['optimized_score'], alpha=0.5)
    plt.plot([df_results['original_score'].min(), df_results['original_score'].max()],
             [df_results['original_score'].min(), df_results['original_score'].max()],
             'r--', label='No change')
    plt.xlabel('Original Score')
    plt.ylabel('Optimized Score')
    plt.title('Original vs Optimized Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to: optimization_comparison.png")



if __name__ == "__main__":
    
    # Example 1: Optimize a single spacer
    print("="*70)
    print("EXAMPLE 1: Single Spacer Optimization")
    print("="*70)
    
    example_spacer = "CCATTGCTAGCTAGCTAGCTAGCTAGCTAG"  # Poor score (starts with CC)
    
    print(f"Original spacer: {example_spacer}")
    original = score_cas13b_spacer(example_spacer)
    print(f"Original score: {original['score']}\n")
    
    optimized_variants = generate_optimized_crRNAs(example_spacer, max_mismatches=3)
    
    print(f"Found {len(optimized_variants)} optimal variant(s):\n")
    for i, variant in enumerate(optimized_variants, 1):
        print(f"Variant {i}:")
        print(f"  Optimized spacer: {variant['optimized_spacer']}")
        print(f"  Score: {variant['optimized_score']} (improvement: +{variant['score_improvement']})")
        print(f"  Mutations: {variant['mutations']}")
        print(f"  Prediction: {variant['prediction']}\n")
    
    # Example 2: Optimize entire CSV
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Optimization from CSV")
    print("="*70 + "\n")
    
    # df_optimized = optimize_spacers_csv(
    #     input_csv='spacers_conservation_filtered.csv',
    #     output_csv='spacers_conservation_filtered_optimized.csv',
    #     max_mismatches=3
    # )
    
    df_optimized = optimize_spacers_csv(
        input_csv='/home/mauli/repos/CAS13b_pipeline/per_subtype/top_k_subtype_spacers.csv',
        output_csv='/home/mauli/repos/CAS13b_pipeline/per_subtype/spacers_conservation_filtered_optimized_subtypes.csv',
        max_mismatches=3
    )
    
    # Generate comparison report
    # compare_original_vs_optimized(df_optimized)