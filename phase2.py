import subprocess
import os
from pathlib import Path
from Bio import AlignIO, SeqIO
import numpy as np
import pandas as pd
from collections import Counter


def perform_msa(input_fasta, output_aln, algorithm="clustal", **kwargs):
    """
    Perform multiple sequence alignment on a FASTA file.
    
    Parameters
    ----------
    input_fasta : str
        Path to input FASTA file with cleaned sequences
    output_aln : str
        Path to output alignment file (format depends on algorithm)
    algorithm : str, default="clustal"
        Alignment algorithm to use: "clustal", "mafft", or "muscle"
    **kwargs : dict
        Additional arguments:
        - clustal_params (str): Additional parameters for Clustal Omega
        - mafft_params (str): Additional parameters for MAFFT
        - muscle_params (str): Additional parameters for MUSCLE
        - output_format (str): Output format ("fasta", "clustal", "phylip"). Default: "fasta"
    
    Returns
    -------
    str
        Path to the output alignment file
    
    Raises
    ------
    ValueError
        If algorithm is not supported or tool not found
    FileNotFoundError
        If input FASTA file doesn't exist
    """
    
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")
    
    output_format = kwargs.get("output_format", "fasta")
    
    if algorithm.lower() == "clustal":
        """Clustal Omega is generally recommended for large alignments"""
        params = kwargs.get("clustal_params", "")
        cmd = f"clustalo -i {input_fasta} -o {output_aln} --outfmt={output_format} {params}"
        
    elif algorithm.lower() == "mafft":
        """MAFFT is faster and good for viral sequences"""
        params = kwargs.get("mafft_params", "--auto")
        cmd = f"mafft {params} {input_fasta} > {output_aln}"
        
    elif algorithm.lower() == "muscle":
        """MUSCLE is also robust, slightly older but reliable"""
        params = kwargs.get("muscle_params", "")
        cmd = f"muscle -in {input_fasta} -out {output_aln} {params}"
        
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from 'clustal', 'mafft', or 'muscle'")
    
    print(f"Running alignment with {algorithm}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ Alignment completed. Output saved to: {output_aln}")
        return output_aln
    except subprocess.CalledProcessError as e:
        print(f"✗ Alignment failed with error:\n{e.stderr}")
        raise RuntimeError(f"MSA failed: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Algorithm '{algorithm}' not found. Please install it first.")


def calculate_conservation_scores(msa_file, output_csv=None, aln_format="fasta", min_gap_threshold=0.5):
    """
    Calculate position-wise conservation scores from a multiple sequence alignment.
    
    Parameters
    ----------
    msa_file : str
        Path to multiple sequence alignment file
    output_csv : str, optional
        Path to save conservation scores as CSV. If None, not saved.
    aln_format : str, default="fasta"
        Format of alignment file ("fasta", "clustal", "phylip", etc.)
    min_gap_threshold : float, default=0.5
        Threshold for gap percentage. Positions with >50% gaps are marked as low confidence.
        Range: 0 to 1
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'positions': List of position numbers (1-indexed)
        - 'consensus': Consensus nucleotide at each position
        - 'conservation_score': Conservation % at each position
        - 'gap_percentage': Gap % at each position
        - 'nucleotide_counts': Counter of nucleotides at each position
        - 'alignment': Biopython AlignIO object
    
    Examples
    --------
    >>> scores = calculate_conservation_scores('alignment.fasta')
    >>> df = pd.DataFrame({
    ...     'Position': scores['positions'],
    ...     'Consensus': scores['consensus'],
    ...     'Conservation': scores['conservation_score'],
    ...     'Gaps': scores['gap_percentage']
    ... })
    >>> df.to_csv('conservation.csv', index=False)
    """
    
    if not os.path.exists(msa_file):
        raise FileNotFoundError(f"MSA file not found: {msa_file}")
    
    print(f"Reading alignment from: {msa_file}")
    
    try:
        alignment = AlignIO.read(msa_file, aln_format)
    except Exception as e:
        raise ValueError(f"Failed to read alignment file. Check format is '{aln_format}': {e}")
    
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    
    print(f"  • Sequences: {num_sequences}")
    print(f"  • Alignment length: {alignment_length} bp")
    
    positions = []
    consensus = []
    conservation_scores = []
    gap_percentages = []
    nucleotide_counts_list = []
    
    # Iterate through each position in the alignment
    for pos in range(alignment_length):
        # Extract nucleotide at this position from all sequences
        nucleotides_at_pos = [str(seq[pos]).upper() for seq in alignment]
        
        # Count nucleotides and gaps
        nucleotide_counter = Counter(nucleotides_at_pos)
        gap_count = nucleotide_counter.get('-', 0) + nucleotide_counter.get('N', 0)
        gap_pct = (gap_count / num_sequences) * 100
        
        # Calculate conservation: % of most common nucleotide (excluding gaps)
        nucleotides_only = [nt for nt in nucleotides_at_pos if nt not in ['-', 'N']]
        
        if len(nucleotides_only) > 0:
            most_common_nt = Counter(nucleotides_only).most_common(1)[0][0]
            most_common_count = Counter(nucleotides_only).most_common(1)[0][1]
            conservation = (most_common_count / len(nucleotides_only)) * 100
        else:
            most_common_nt = '-'
            conservation = 0.0
        
        positions.append(pos + 1)  # 1-indexed
        consensus.append(most_common_nt)
        conservation_scores.append(conservation)
        gap_percentages.append(gap_pct)
        nucleotide_counts_list.append(nucleotide_counter)
    
    print(f"✓ Conservation scores calculated for {alignment_length} positions")
    
    # Create results dictionary
    results = {
        'positions': positions,
        'consensus': consensus,
        'conservation_score': conservation_scores,
        'gap_percentage': gap_percentages,
        'nucleotide_counts': nucleotide_counts_list,
        'alignment': alignment,
        'num_sequences': num_sequences,
        'alignment_length': alignment_length
    }
    
    # Optionally save to CSV
    if output_csv:
        df = pd.DataFrame({
            'Position': positions,
            'Consensus': consensus,
            'Conservation_Score': conservation_scores,
            'Gap_Percentage': gap_percentages
        })
        df.to_csv(output_csv, index=False)
        print(f"✓ Conservation scores saved to: {output_csv}")
    
    return results


def extract_alignment_stats(conservation_results):
    """
    Extract and print summary statistics from conservation calculations.
    
    Parameters
    ----------
    conservation_results : dict
        Output from calculate_conservation_scores()
    
    Returns
    -------
    dict
        Summary statistics
    """
    
    conservation_scores = conservation_results['conservation_score']
    gap_percentages = conservation_results['gap_percentage']
    
    stats = {
        'mean_conservation': np.mean(conservation_scores),
        'median_conservation': np.median(conservation_scores),
        'min_conservation': np.min(conservation_scores),
        'max_conservation': np.max(conservation_scores),
        'std_conservation': np.std(conservation_scores),
        'mean_gap_pct': np.mean(gap_percentages),
        'high_conservation_positions': sum(1 for cs in conservation_scores if cs >= 90),  # >=90% conserved
        'highly_conserved_regions': []  # To be filled next
    }
    
    # Find highly conserved regions (>80% conservation, >5 consecutive positions)
    conserved_region = []
    for i, cs in enumerate(conservation_scores):
        if cs >= 80:
            if conserved_region and i == conserved_region[-1] + 1:
                conserved_region.append(i)
            elif conserved_region:
                if len(conserved_region) >= 5:
                    stats['highly_conserved_regions'].append((conserved_region[0] + 1, conserved_region[-1] + 1))
                conserved_region = [i]
            else:
                conserved_region = [i]
    if len(conserved_region) >= 5:
        stats['highly_conserved_regions'].append((conserved_region[0] + 1, conserved_region[-1] + 1))
    
    print("\n" + "="*60)
    print("CONSERVATION STATISTICS")
    print("="*60)
    print(f"Mean conservation:          {stats['mean_conservation']:.2f}%")
    print(f"Median conservation:        {stats['median_conservation']:.2f}%")
    print(f"Range:                      {stats['min_conservation']:.2f}% - {stats['max_conservation']:.2f}%")
    print(f"Std deviation:              {stats['std_conservation']:.2f}%")
    print(f"Positions ≥90% conserved:   {stats['high_conservation_positions']}")
    print(f"Mean gap percentage:        {stats['mean_gap_pct']:.2f}%")
    print(f"Highly conserved regions (≥80%, ≥5 nt): {len(stats['highly_conserved_regions'])}")
    
    if stats['highly_conserved_regions']:
        print("\nConserved regions (start-end positions):")
        for start, end in stats['highly_conserved_regions'][:10]:  # Print first 10
            print(f"  • Positions {start}-{end} ({end-start+1} nt)")
    
    print("="*60 + "\n")
    
    return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Paths (modify these to match your data)
    input_fasta = "cleaned_dengue_genomes.fasta"
    output_alignment = "dengue_alignment.fasta"
    output_conservation_csv = "conservation_scores.csv"
    
    # Step 2.1: Perform Multiple Sequence Alignment
    # Choose algorithm based on your needs:
    # - "clustal": Most reliable, good for small-medium datasets, slower
    # - "mafft": Fast, scalable, good for large datasets
    # - "muscle": Good balance, older but robust
    
    msa_file = perform_msa(
        input_fasta=input_fasta,
        output_aln=output_alignment,
        algorithm="mafft",  # Change this to try different algorithms
        mafft_params="--auto",  # Auto-detect best strategy
        output_format="fasta"
    )
    
    # Step 2.2: Calculate Conservation Scores
    conservation_results = calculate_conservation_scores(
        msa_file=msa_file,
        output_csv=output_conservation_csv,
        aln_format="fasta",
        min_gap_threshold=0.5
    )
    
    # Extract and display statistics
    stats = extract_alignment_stats(conservation_results)
    
    # Optional: Create a more detailed report
    df_conservation = pd.DataFrame({
        'Position': conservation_results['positions'],
        'Consensus': conservation_results['consensus'],
        'Conservation': conservation_results['conservation_score'],
        'Gaps': conservation_results['gap_percentage']
    })
    
    print(f"\nFirst 20 positions of alignment:")
    print(df_conservation.head(20).to_string(index=False))
