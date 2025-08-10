import pandas as pd
import numpy as np
import os
import collections
from tqdm import tqdm

def analyze_data(sampling_files, output_prefix="results"):
    """
    Analyze sampling results from multiple embryos.
    This function generates individual and concatenated tissue-barcode matrices,
    and calculates fate coupling for individual embryos and the combined dataset.

    Args:
        sampling_files (list): A list of paths to the sampling data CSV files.
        output_prefix (str): The prefix for all output files.
    """
    # --- Step 1: Check for required packages ---
    try:
        import anndata as ad
        import cospar as cs
    except ImportError:
        print("Error: This analysis requires anndata and cospar packages.")
        print("Please install them with: pip install anndata cospar")
        return

    # --- Step 2: Define tissue name mapping ---
    tissue_names = {
        16: "blood", 17: "L brain I", 18: "R brain I", 19: "L brain III", 
        20: "R brain III", 21: "L gonad", 22: "R gonad", 23: "L kidney", 
        24: "R kidney", 25: "L foot", 26: "L leg", 27: "R foot", 
        28: "R leg", 29: "L hand", 30: "L arm", 31: "R hand", 32: "R arm"
    }

    # --- Step 3: Process each embryo file ---
    all_barcode_matrices = []  # To store matrices for final concatenation

    for embryo_idx, sampling_file in enumerate(tqdm(sampling_files, desc="Processing Embryos")):
        embryo_num = embryo_idx + 1
        print(f"\nProcessing {sampling_file}...")

        # Read and parse the data for one embryo
        embryo_data = []
        with open(sampling_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2: continue
                
                loxcode = tuple(map(int, parts[0].strip().split()))
                fate_parts = parts[-1].strip().split()
                if fate_parts:
                    tissue_type = int(fate_parts[-1])
                    if tissue_type >= 16:
                        embryo_data.append((loxcode, tissue_type))

        if not embryo_data:
            print(f"Warning: No valid data found in {sampling_file}. Skipping.")
            continue

        # Create a tissue-barcode count matrix for the current embryo
        tissue_barcode_counts = {tissue_id: collections.Counter() for tissue_id in range(16, 33)}
        for barcode, tissue_id in embryo_data:
            tissue_barcode_counts[tissue_id][barcode] += 1

        all_barcodes = sorted(list(set(bc for _, counter in tissue_barcode_counts.items() for bc in counter.keys())))
        if not all_barcodes:
            print(f"Warning: No barcodes found for embryo #{embryo_num}. Skipping.")
            continue

        # Build the DataFrame
        matrix_data = {
            tissue_names[tid]: [tissue_barcode_counts[tid][bc] for bc in all_barcodes]
            for tid in sorted(tissue_barcode_counts.keys())
        }
        barcode_labels = [f"embryo{embryo_num}_{str(b)}" for b in all_barcodes]
        barcode_matrix = pd.DataFrame(matrix_data, index=barcode_labels).T
        
        # Save the individual tissue-barcode matrix
        barcode_matrix.to_csv(f"{output_prefix}_embryo{embryo_num}_tissue_barcode_matrix.csv")
        print(f"Saved individual tissue-barcode matrix for embryo #{embryo_num}")
        all_barcode_matrices.append(barcode_matrix)

        # --- Step 4: Calculate and save individual fate coupling ---
        try:
            # Create AnnData object for the individual embryo
            adata_individual = ad.AnnData(barcode_matrix)
            adata_individual.obs['state_info'] = barcode_matrix.index
            # FIX: Add dummy time_info required by cospar
            adata_individual.obs['time_info'] = 1.0
            adata_individual.obsm["X_clone"] = barcode_matrix.values
            
            # Calculate fate coupling
            cs.tl.fate_coupling(adata_individual, source="X_clone")
            
            # Extract and save the coupling matrix
            individual_coupling = pd.DataFrame(
                adata_individual.uns['fate_coupling_X_clone']['X_coupling'],
                index=adata_individual.uns['fate_coupling_X_clone']['fate_names'],
                columns=adata_individual.uns['fate_coupling_X_clone']['fate_names']
            )
            individual_coupling.to_csv(f"{output_prefix}_embryo{embryo_num}_fate_coupling.csv")
            print(f"Saved individual fate coupling matrix for embryo #{embryo_num}")

        except Exception as e:
            print(f"Warning: Could not calculate fate coupling for embryo #{embryo_num}: {e}")

    # --- Step 5: Concatenate all matrices and calculate final MCSA matrix ---
    if not all_barcode_matrices:
        print("No data processed. Exiting analysis.")
        return

    print("\nConcatenating all embryo matrices...")
    concatenated_matrix = pd.concat(all_barcode_matrices, axis=1)
    concatenated_matrix.to_csv(f"{output_prefix}_concatenated_tissue_barcode_matrix.csv")
    print(f"Saved concatenated matrix with shape: {concatenated_matrix.shape}")

    print("Calculating final MCSA matrix on concatenated data...")
    try:
        # Create AnnData object from the concatenated matrix
        adata_concat = ad.AnnData(concatenated_matrix)
        adata_concat.obs['state_info'] = concatenated_matrix.index
        # FIX: Add dummy time_info required by cospar
        adata_concat.obs['time_info'] = 1.0
        adata_concat.obsm["X_clone"] = concatenated_matrix.values

        # Calculate fate coupling on the combined data
        cs.tl.fate_coupling(adata_concat, source="X_clone")

        # Extract and save the final MCSA matrix
        mcsa_matrix = pd.DataFrame(
            adata_concat.uns['fate_coupling_X_clone']['X_coupling'],
            index=adata_concat.uns['fate_coupling_X_clone']['fate_names'],
            columns=adata_concat.uns['fate_coupling_X_clone']['fate_names']
        )
        mcsa_matrix.to_csv(f"{output_prefix}_mcsa_matrix.csv")
        print("Successfully saved the final MCSA matrix.")

    except Exception as e:
        print(f"Error: Could not calculate the final MCSA matrix: {e}")


if __name__ == "__main__":
    # Define the input files
    # Assumes these files are in the same directory as the script
    input_files = [
        "loxcode_census_at_sampling_embryo1.csv",
        "loxcode_census_at_sampling_embryo2.csv",
        "loxcode_census_at_sampling_embryo3.csv"
    ]

    # Check if input files exist
    valid_files = [f for f in input_files if os.path.exists(f)]
    if not valid_files:
        print("Error: None of the required input files were found.")
        print("Please ensure the following files are in the script's directory:")
        for f in input_files:
            print(f" - {f}")
    else:
        # Run the analysis
        analyze_data(valid_files, output_prefix="results")