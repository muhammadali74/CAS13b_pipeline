import sqlite3
from Bio import SeqIO
from tqdm import tqdm

def extract_spacers_sqlite(fasta_path, spacer_length=30, db_path='spacers.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create optimized schema
    cur.execute("CREATE TABLE IF NOT EXISTS spacers (id INTEGER PRIMARY KEY, seq TEXT UNIQUE)")
    cur.execute("CREATE TABLE IF NOT EXISTS genome_map (spacer_id INTEGER, genome_id TEXT, UNIQUE(spacer_id, genome_id))")

    conn.commit()

    # Count sequences for tqdm
    total = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))

    # Process each genome
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), total=total):
        seq = str(record.seq).upper()
        genome_id = record.id

        # Unique spacers per genome
        spacers = set()
        for i in range(len(seq) - spacer_length + 1):
            s = seq[i:i+spacer_length]
            if "N" in s or "-" in s:
                continue
            spacers.add(s)

        # Batch insert
        for s in spacers:
            # Insert spacer (or ignore if exists)
            cur.execute("INSERT OR IGNORE INTO spacers (seq) VALUES (?)", (s,))
            # Fetch its id
            cur.execute("SELECT id FROM spacers WHERE seq = ?", (s,))
            sid = cur.fetchone()[0]
            # Insert mapping (avoid duplicates)
            cur.execute("INSERT OR IGNORE INTO genome_map VALUES (?, ?)", (sid, genome_id))

        conn.commit()  # commit once per genome (fast)

    conn.close()


import sqlite3
import pandas as pd
from tqdm import tqdm

def spacers_to_dataframe_sqlite(db_path, total_genomes, batch_size=100000, output_csv='spacers_summary.csv'):
    """
    Convert SQLite spacer database into CSV in batches — memory efficient.

    db_path       : Path to SQLite database.
    total_genomes : Total number of genomes.
    batch_size    : Number of spacers processed per batch.
    output_csv    : Output CSV file (written incrementally).
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Count spacers
    cur.execute("SELECT COUNT(*) FROM spacers")
    total_spacers = cur.fetchone()[0]

    # Prepare CSV file
    header_written = False

    # Process in batches
    for offset in tqdm(range(0, total_spacers, batch_size)):
        # 1. Fetch batch of spacer entries
        cur.execute("""
            SELECT id, seq
            FROM spacers
            ORDER BY id
            LIMIT ? OFFSET ?
        """, (batch_size, offset))
        
        rows = cur.fetchall()

        batch_data = []

        for spacer_id, spacer_seq in rows:
            # 2. Fetch genome IDs for this spacer
            cur.execute("""
                SELECT genome_id
                FROM genome_map
                WHERE spacer_id = ?
            """, (spacer_id,))
            genome_ids = [row[0] for row in cur.fetchall()]

            n_genomes = len(genome_ids)
            conservation_pct = 100 * n_genomes / total_genomes

            batch_data.append({
                "spacer": spacer_seq,
                "n_genomes": n_genomes,
                "conservation_pct": conservation_pct,
                "genome_ids": ",".join(genome_ids)
            })

        # 3. Write to CSV incrementally
        df = pd.DataFrame(batch_data)
        df.to_csv(output_csv, mode='a', header=not header_written, index=False)
        header_written = True  # only write once

    conn.close()


if __name__ == "__main__":
    fasta_path = "filtered_dengue_genomes.fasta"
    spacer_length = 30
    db_path = 'spacers.db'

    # 1. Extract spacers using SQLite version
    extract_spacers_sqlite(fasta_path, spacer_length, db_path)

    # 2. Count unique genomes
    total_genomes = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))

    # 3. Convert SQLite DB → CSV in batches (efficient)
    spacers_to_dataframe_sqlite(
        db_path,
        total_genomes,
        batch_size=50000,
        output_csv='spacers_conservation.csv'
    )
