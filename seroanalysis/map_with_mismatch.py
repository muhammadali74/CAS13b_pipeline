#!/usr/bin/env python3
"""
Map Cas13b spacer sequences to Dengue virus genome with mismatch tolerance.

Usage:
    python map_spacer_to_genome_fuzzy.py <genome.fasta> <spacers.fasta> <max_mismatches>

Example:
    python map_spacer_to_genome_fuzzy.py DENV1.fasta spacers.fasta 2
"""

import sys
from Bio import SeqIO
from Bio.Seq import Seq

def reverse_complement(seq_str):
    """Return reverse complement of a DNA/RNA sequence."""
    seq = Seq(seq_str.upper().replace('U', 'T'))
    return str(seq.reverse_complement())

def count_mismatches(seq1, seq2):
    """Count mismatches between two equal-length sequences."""
    return sum(1 for a, b in zip(seq1, seq2) if a != b)

def find_matches_with_mismatches(target_seq, query_seq, max_mismatches=2):
    """
    Find all matches of query_seq in target_seq allowing up to max_mismatches.
    Returns list of tuples: (start_pos, end_pos, strand, match_type, num_mismatches)
    """
    matches = []
    target = str(target_seq).upper()
    query = query_seq.upper().replace('U', 'T')
    query_len = len(query)
    
    # Search forward strand
    for i in range(len(target) - query_len + 1):
        window = target[i:i+query_len]
        mismatches = count_mismatches(query, window)
        if mismatches <= max_mismatches:
            matches.append((i, i + query_len, '+', 'direct', mismatches))
            print(window)
    
    # Search reverse strand
    query_rc = reverse_complement(query)
    for i in range(len(target) - query_len + 1):
        window = target[i:i+query_len]
        mismatches = count_mismatches(query_rc, window)
        if mismatches <= max_mismatches:
            matches.append((i, i + query_len, '-', 'reverse_complement', mismatches))
    
    return matches

def main(genome_fasta, spacers_fasta, max_mismatches):
    # Load genome
    print(f"Loading genome from {genome_fasta}...", file=sys.stderr)
    genome_records = list(SeqIO.parse(genome_fasta, "fasta"))
    if len(genome_records) == 0:
        print("Error: No genome sequences found.", file=sys.stderr)
        sys.exit(1)
    
    genome_record = genome_records[0]
    genome_id = genome_record.id
    genome_seq = genome_record.seq
    print(f"Genome ID: {genome_id}, Length: {len(genome_seq)} bp", file=sys.stderr)
    
    # Load spacers
    print(f"Loading spacers from {spacers_fasta}...", file=sys.stderr)
    spacers = list(SeqIO.parse(spacers_fasta, "fasta"))
    print(f"Found {len(spacers)} spacers", file=sys.stderr)
    print(f"Allowing up to {max_mismatches} mismatches", file=sys.stderr)
    
    # Output header
    print("spacer_name,spacer_seq,genome_id,nt_start,nt_end,strand,match_type,num_mismatches")
    
    # Map each spacer
    total_mapped = 0
    spacers_with_matches = 0
    
    for spacer in spacers:
        spacer_name = spacer.id
        spacer_seq = str(spacer.seq).upper()
        
        matches = find_matches_with_mismatches(genome_seq, spacer_seq, max_mismatches)
        
        if len(matches) == 0:
            # No match found
            print(f"{spacer_name},{spacer_seq},{genome_id},NA,NA,NA,no_match,NA")
        else:
            spacers_with_matches += 1
            # Sort by number of mismatches (best matches first)
            matches.sort(key=lambda x: x[4])
            
            for (start, end, strand, match_type, num_mm) in matches:
                # Convert to 1-based coordinates
                start_1based = start + 1
                end_1based = end
                print(f"{spacer_name},{spacer_seq},{genome_id},{start_1based},{end_1based},{strand},{match_type},{num_mm}")
                total_mapped += 1
    
    print(f"\nTotal matches found: {total_mapped}", file=sys.stderr)
    print(f"Spacers with at least one match: {spacers_with_matches}/{len(spacers)}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python map_spacer_to_genome_fuzzy.py <genome.fasta> <spacers.fasta> <max_mismatches>", file=sys.stderr)
        print("Example: python map_spacer_to_genome_fuzzy.py DENV1.fasta spacers.fasta 2", file=sys.stderr)
        sys.exit(1)
    
    genome_fasta = sys.argv[1]
    spacers_fasta = sys.argv[2]
    max_mismatches = int(sys.argv[3])
    
    main(genome_fasta, spacers_fasta, max_mismatches)