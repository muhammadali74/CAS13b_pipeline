#!/usr/bin/env python3
"""
Map Cas13b spacer sequences to Dengue virus genome and report coordinates.

For Cas13b:
- The spacer is complementary to the target RNA
- Search for BOTH the spacer sequence AND its reverse complement in the genome
- This accounts for whether the target is on sense or antisense strand

Usage:
    python map_spacer_to_genome.py <genome.fasta> <spacers.fasta>

Output:
    CSV with columns: spacer_name, spacer_seq, genome_id, nt_start, nt_end, strand, match_type
"""

import sys
from Bio import SeqIO
from Bio.Seq import Seq

def reverse_complement(seq_str):
    """Return reverse complement of a DNA/RNA sequence."""
    seq = Seq(seq_str.upper().replace('U', 'T'))
    return str(seq.reverse_complement())

def find_all_matches(target_seq, query_seq):
    """
    Find all exact matches of query_seq in target_seq.
    Returns list of tuples: (start_pos, end_pos, strand)
    start_pos is 0-based
    """
    matches = []
    target = str(target_seq).upper()
    query = query_seq.upper()
    
    # Search forward strand (spacer matches genome directly)
    start = 0
    while True:
        pos = target.find(query, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(query), '+', 'direct'))
        start = pos + 1
    
    # Search reverse strand (spacer's reverse complement matches genome)
    # This means the target RNA is on the opposite strand
    query_rc = reverse_complement(query)
    start = 0
    while True:
        pos = target.find(query_rc, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(query), '-', 'reverse_complement'))
        start = pos + 1
    
    return matches

def main(genome_fasta, spacers_fasta):
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
    
    # Output header
    print("spacer_name,spacer_seq,genome_id,nt_start,nt_end,strand,match_type")
    
    # Map each spacer
    total_mapped = 0
    for spacer in spacers:
        spacer_name = spacer.id
        spacer_seq = str(spacer.seq).upper()
        
        matches = find_all_matches(genome_seq, spacer_seq)
        
        if len(matches) == 0:
            # No match found
            print(f"{spacer_name},{spacer_seq},{genome_id},NA,NA,NA,no_match")
        else:
            for (start, end, strand, match_type) in matches:
                # Convert to 1-based coordinates for biological convention
                start_1based = start + 1
                end_1based = end
                print(f"{spacer_name},{spacer_seq},{genome_id},{start_1based},{end_1based},{strand},{match_type}")
                total_mapped += 1
    
    print(f"\nTotal matches found: {total_mapped}", file=sys.stderr)
    print(f"Spacers with at least one match: {sum(1 for s in spacers if len(find_all_matches(genome_seq, str(s.seq))) > 0)}/{len(spacers)}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python map_spacer_to_genome.py <genome.fasta> <spacers.fasta>", file=sys.stderr)
        sys.exit(1)
    
    genome_fasta = sys.argv[1]
    spacers_fasta = sys.argv[2]
    
    main(genome_fasta, spacers_fasta)