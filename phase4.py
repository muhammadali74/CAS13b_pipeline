import pandas as pd
import re

def calculate_gc_content(seq):
    """Calculate GC content percentage for a DNA/RNA sequence."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return 100 * gc_count / len(seq) if len(seq) > 0 else 0

def has_homopolymer(seq, run=4):
    """Detect homopolymeric runs of specified length (default: 4)."""
    for base in 'ATGC':
        if base * run in seq.upper():
            return True
    return False

def annotate_spacers(input_csv, output_csv, gc_min=40, gc_max=60, homopolymer_run=4, conservation_threshold=15):
    """
    Load CSV of spacers, add GC content and homopolymer columns, and filter.
    Save results to new CSVs.
    """
    df = pd.read_csv(input_csv)
    
    # Phase 4.1: GC content annotation
    # df['GC_content'] = df['spacer'].apply(calculate_gc_content)
    
    # # Phase 4.2: Homopolymeric run annotation
    # df['Has_homopolymer'] = df['spacer'].apply(lambda x: has_homopolymer(x, run=homopolymer_run))
    
    # # Save annotated full table
    # df.to_csv(output_csv.replace('.csv', '_annotated.csv'), index=False)
    
    # Phase 4.3: Filtering
    filtered_df = df[
        # (df['GC_content'] >= gc_min) &
        # (df['GC_content'] <= gc_max) &
        (df['conservation_pct'] >= conservation_threshold)
        # (~df['Has_homopolymer'])
    ].reset_index(drop=True)
    
    # Save filtered table
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtering complete. Annotated CSV: {output_csv.replace('.csv', '_annotated.csv')}")
    print(f"Filtered CSV: {output_csv}")
    print(f"Spacers retained: {len(filtered_df)} / {len(df)}")

# Example usage:
if __name__ == "__main__":
    input_csv = "spacers_conservation_filtered_annotated.csv"
    output_csv = "new/spacers_conservation_filtered.csv"
    annotate_spacers(input_csv, output_csv)
