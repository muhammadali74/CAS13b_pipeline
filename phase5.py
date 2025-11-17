import pandas as pd
import subprocess
import tempfile
import os

def run_rnafold(sequence):
    """
    Run RNAfold on a single RNA sequence. Returns (structure, mfe).
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(mode="r", delete=False) as temp_out:
        temp_in.write(sequence + '\n')
        temp_in.flush()
        temp_in.close()
        cmd = f'RNAfold < {temp_in.name}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            os.remove(temp_in.name)
            return ("", None)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            os.remove(temp_in.name)
            return ("", None)
        structure_line = lines[1]
        # Parse structure and free energy, e.g. ".((((...))))... (-2.50)"
        if '(' in structure_line or ')' in structure_line:
            structure, mfe_str = structure_line.rsplit(' ', 1)
            mfe = float(mfe_str.replace('(', '').replace(')', ''))
        else:
            structure = ""
            mfe = None
        os.remove(temp_in.name)
        return (structure, mfe)

def annotate_rnafold(csv_in, csv_out):
    """
    Annotate spacers in a CSV with RNA secondary structure and MFE score.
    """
    df = pd.read_csv(csv_in)
    structures = []
    mfes = []
    print("Running RNAfold for all spacers. This may take a while...")
    for seq in df['spacer']:
        structure, mfe = run_rnafold(seq)
        print(f"Processed sequence: {seq[:30]}... -> Structure: {structure}, MFE: {mfe}")
        structures.append(structure)
        mfes.append(mfe)
    df['rna_structure'] = structures
    df['mfe_stability'] = mfes
    df.to_csv(csv_out, index=False)
    print(f"Annotated CSV with RNAfold results saved as: {csv_out}")

# Example usage
if __name__ == "__main__":
    input_csv = "spacers_conservation_filtered.csv"
    output_csv = "spacers_conservation_filtered_struct.csv"
    annotate_rnafold(input_csv, output_csv)