import pandas as pd


B = pd.read_csv("/home/mauli/repos/CAS13b_pipeline/BVBRC_genome_sequences.csv")

# Normalize
B["Genome ID"] = B["Genome ID"].astype(str).str.strip()

# Subtype classification
def classify_subtype(name):
    name = str(name).lower()
    if " 1" in name:
        return "type1"
    elif " 2" in name:
        return "type2"
    elif " 3" in name:
        return "type3"
    elif " 4" in name:
        return "type4"
    else:
        return "unclassified"

B["subtype"] = B["Genome Name"].apply(classify_subtype)

# Build dictionary for fast lookup
subtype_map = B.set_index("Genome ID")["subtype"].to_dict()


input_csv = "/home/mauli/repos/CAS13b_pipeline/spacers_conservation_filtered_annotated.csv"
output_csv = "spacers_conservation_filtered_annotated_with_subtypes.csv"

chunksize = 50_000   # adjust depending on RAM

first_write = True   # to control header writing

for chunk in pd.read_csv(input_csv, chunksize=chunksize):

 
    exploded = (
        chunk.assign(genome_ids=chunk["genome_ids"].str.split(","))
             .explode("genome_ids")
    )
    exploded["genome_ids"] = exploded["genome_ids"].str.strip()

    # Map subtype
    exploded["subtype"] = exploded["genome_ids"].map(subtype_map).fillna("unclassified")


    subtype_counts = (
        exploded.groupby(exploded.index)["subtype"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # Ensure all subtype columns always exist
    for st in ["type1", "type2", "type3", "type4", "unclassified"]:
        if st not in subtype_counts.columns:
            subtype_counts[st] = 0

    # Merge counts back into chunk
    merged = chunk.join(subtype_counts, how="left")

    # Replace NaNs with 0
    merged[["type1", "type2", "type3", "type4", "unclassified"]] = \
        merged[["type1", "type2", "type3", "type4", "unclassified"]].fillna(0).astype(int)


    merged.to_csv(
        output_csv, 
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    first_write = False

    print(f"Processed {len(chunk)} rows...")

print("\nDone! Output saved to", output_csv)
