from Bio import SeqIO
from collections import defaultdict
import pandas as pd

def extract_spacers_and_conservation(fasta_path, spacer_length=30):
    """
    Extract all unique spacers of given length from a FASTA file of genomes.
    Track in how many unique genomes each spacer appears.
    
    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file (one or more genomes).
    spacer_length : int, default=30
        Length of spacer to extract (Cas13b default is 30 nt).
    
    Returns
    -------
    dict
        Dictionary with spacer sequences as keys,
        and set of genome IDs in which they appear as values.
    """
    # spacer_dict has format: spacer_seq -> set of genome_ids
    spacer_dict = defaultdict(set)
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        genome_id = record.id
        # Use a set to avoid double-counting spacers within the same genome
        spacers_in_this_genome = set()
        for i in range(len(seq) - spacer_length + 1):
            spacer = seq[i:i+spacer_length]
            if "N" in spacer or "-" in spacer:
                continue  # Skip ambiguous or gapped spacers
            spacers_in_this_genome.add(spacer)
        for spacer in spacers_in_this_genome:
            spacer_dict[spacer].add(genome_id)
    
    return spacer_dict  


from Bio import SeqIO
import shelve

def extract_spacers_disk(fasta_path, spacer_length=30, db_path='spacers.db'):
    db = shelve.open(db_path, flag='c')  # disk-backed dictionary
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        genome_id = record.id
        spacers_in_this_genome = set()
        for i in range(len(seq) - spacer_length + 1):
            spacer = seq[i:i+spacer_length]
            if "N" in spacer or "-" in spacer:
                continue
            spacers_in_this_genome.add(spacer)
        for spacer in spacers_in_this_genome:
            genomes = db.get(spacer, set())
            genomes.add(genome_id)
            db[spacer] = genomes
    db.close()
    # To read: db = shelve.open(db_path)


def spacers_to_dataframe(spacer_dict, total_genomes):
    """
    Convert spacer dictionary to pandas DataFrame with conservation info.
    
    Parameters
    ----------
    spacer_dict : dict
        Output of extract_spacers_and_conservation.
    total_genomes : int
        Total number of unique input genomes.
    
    Returns
    -------
    pandas.DataFrame
        Columns: spacer, n_genomes, conservation_pct, genome_ids
    """
    import pandas as pd
    spacer_list = []
    for spacer, genome_ids in spacer_dict.items():
        n_genomes = len(genome_ids)
        conservation_pct = 100 * n_genomes / total_genomes
        spacer_list.append({
            "spacer": spacer,
            "n_genomes": n_genomes,
            "conservation_pct": conservation_pct,
            "genome_ids": ",".join(genome_ids)
        })
    df = pd.DataFrame(spacer_list)
    return df


def spacers_to_dataframe_eff(db_path, total_genomes, batch_size=100000, output_csv='spacers_summary.csv'):
    """
    Convert disk-backed spacer database to pandas DataFrame in batches,
    avoiding out-of-memory issues.
    
    Parameters
    ----------
    db_path : str
        Path to shelve (disk-backed) spacer db.
    total_genomes : int
        Count of unique input genomes.
    batch_size : int
        Number of spacers processed per batch (default: 100k).
    output_csv : str
        Output file for results (written incrementally).
    
    Returns
    -------
    None
        Writes results directly to CSV to avoid keeping full DataFrame in memory.
    """
    db = shelve.open(db_path, flag='r')
    spacer_list = []
    count = 0
    for spacer in db:
        genome_ids = db[spacer]
        n_genomes = len(genome_ids)
        conservation_pct = 100 * n_genomes / total_genomes
        spacer_list.append({
            "spacer": spacer,
            "n_genomes": n_genomes,
            "conservation_pct": conservation_pct,
            "genome_ids": ",".join(genome_ids)
        })
        count += 1
        if count % batch_size == 0:
            pd.DataFrame(spacer_list).to_csv(output_csv, mode='a', header=(count == batch_size), index=False)
            spacer_list = []
    # Write any remaining rows
    if spacer_list:
        pd.DataFrame(spacer_list).to_csv(output_csv, mode='a', header=(count < batch_size), index=False)
    db.close()


# ==== Usage Example ====
if __name__ == "__main__":
    fasta_path = "filtered_dengue_genomes.fasta"
    spacer_length = 30
    db_path = 'spacers.db'

    # 1. Extract spacers from FASTA to disk (RAM efficient)
    extract_spacers_disk(fasta_path, spacer_length, db_path)  # From previous response

    # 2. Get total genome count
    genome_ids = set()
    for record in SeqIO.parse(fasta_path, "fasta"):
        genome_ids.add(record.id)
    total_genomes = len(genome_ids)

    # 3. Convert disk-based spacers to dataframe/batch CSV
    spacers_to_dataframe_eff(db_path, total_genomes, batch_size=100000, output_csv='spacers_conservation.csv')
