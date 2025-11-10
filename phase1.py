import pandas as pd

def filter_and_convert_to_fasta(
    csv_file, fasta_output,
    ambiguous_threshold=5.0,  # percentage (e.g., 5%)
    min_length=1000,          # minimum sequence length
    seq_col="Sequence",       # column name containing genome sequences
    id_col="ID"               # optional column for labeling FASTA entries
):
    """
    Filters genome sequences based on ambiguity and length,
    and writes valid entries to a FASTA file.
    """
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found in CSV file.")
    
    # Generate a unique identifier if no ID column exists
    if id_col not in df.columns:
        df[id_col] = [f"seq_{i+1}" for i in range(len(df))]
    
    # Function to calculate percentage of ambiguous bases (N)
    def ambiguity_percentage(seq):
        seq = str(seq).upper()
        if len(seq) == 0:
            return 100.0
        n_count = seq.count('N')
        return (n_count / len(seq)) * 100
    
    # Apply filters
    df["ambiguous_pct"] = df[seq_col].apply(ambiguity_percentage)
    df["length"] = df[seq_col].apply(lambda s: len(str(s)))
    
    filtered_df = df[
        (df["ambiguous_pct"] <= ambiguous_threshold) &
        (df["length"] >= min_length)
    ]
    
    # Write to FASTA
    with open(fasta_output, "w") as fasta_file:
        for _, row in filtered_df.iterrows():
            header = f">{row[id_col]}"
            # You can include more metadata in the header if desired
            # Example: f">{row[id_col]} | len={row['length']} | amb={row['ambiguous_pct']:.2f}%"
            seq = str(row[seq_col]).replace("\n", "").upper()
            fasta_file.write(f"{header}\n{seq}\n")
    
    print(f"✅ Filtered {len(filtered_df)}/{len(df)} sequences written to '{fasta_output}'.")


# Example usage
if __name__ == "__main__":
    filter_and_convert_to_fasta(
        csv_file="genomes.csv",
        fasta_output="filtered_genomes.fasta",
        ambiguous_threshold=5.0,  # remove sequences with >5% 'N'
        min_length=1000           # remove sequences shorter than 1000 bp
    )
