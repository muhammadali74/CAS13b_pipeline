import pandas as pd

def make_top_k_csv(input_csv, output_csv, k=50, deduplicate=True):
    """
    From a CSV with subtype count columns (type1, type2, type3, type4, unclassified),
    extract the top-k rows per subtype and label them.
    
    Args:
        input_csv: Path to annotated CSV.
        output_csv: Where to save the top-k combined CSV.
        k: Number of rows to pick per subtype.
        deduplicate: Whether to remove duplicate rows that appear in multiple top-k lists.
    """
    
    df = pd.read_csv(input_csv)

    subtype_cols = ["type1", "type2", "type3", "type4", "unclassified"]
    results = []

    for subtype in subtype_cols:
        # Get top-k rows for this subtype
        topk = (
            df.sort_values(by=subtype, ascending=False)
              .head(k)
              .copy()
        )
        topk["top_k_subtype"] = subtype
        results.append(topk)

    # Combine all subtype top-k rows
    combined = pd.concat(results, ignore_index=True)

    # Optionally remove duplicate rows that appear in multiple subtype top-k lists
    if deduplicate:
        combined = combined.drop_duplicates(subset=df.columns.tolist(), keep="first")

    combined.to_csv(output_csv, index=False)

    print(f"Done. Top-k CSV written to: {output_csv}")
    print(f"Rows in final output: {len(combined)}")


if __name__ == "__main__":
    make_top_k_csv(
        input_csv="/home/mauli/repos/CAS13b_pipeline/spacers_conservation_filtered_annotated_with_subtypes.csv",
        output_csv="top_k_subtype_spacers.csv",
        k=300  # change as needed
    )
