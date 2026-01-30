
import os
import argparse
import pandas as pd
import numpy as np

def find_and_analyze_csvs(root_dir): 
    """
    Recursively finds 'summary_comparison.csv' files, excluding 'projection' folders,
    and analyzes them for best values.
    """
    summary_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to exclude 'projection' folders
        dirs[:] = [d for d in dirs if 'projection' not in d.lower()]
        
        if "summary_comparison.csv" in files:
            summary_files.append(os.path.join(root, "summary_comparison.csv"))

    print(f"Found {len(summary_files)} summary_comparison.csv files.")
    print("-" * 80)

    all_results = []
    for file_path in summary_files:
        analyze_file(file_path, all_results)

    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(root_dir, "batch_analysis_results.csv")
        try:
            results_df.to_csv(output_path, index=False)
            print(f"\nBatch analysis results saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving batch results to CSV: {e}")
    else:
        print("\nNo results to save.")

def analyze_file(file_path, results_list):
    print(f"\nAnalyzing: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Identify numeric columns, excluding typical metadata or string columns
    # We assume 'filename' and similar are not what we want to maximize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out columns that might be indices or non-metric counters if possible,
    # but for now, we'll take all numeric columns.
    
    if not numeric_cols:
        print("  No numeric columns found.")
        return

    print(f"  Found {len(df)} rows.")

    for col in numeric_cols:
        if df[col].dropna().empty:
            continue
            
        max_val = df[col].max()
        best_row = df.loc[df[col].idxmax()]
        
        # Determine the "winner" label (e.g., filename)
        label_col = 'filename' if 'filename' in df.columns else df.columns[0]
        winner_label = best_row[label_col]

        # Get the rest of the values
        rest = df[df[col] != max_val][col]
        
        print(f"  Column: {col}")
        print(f"    Best Value: {max_val} ({winner_label})")
        
        result_entry = {
            'file_path': file_path,
            'column': col,
            'best_value': max_val,
            'best_label': winner_label,
            'second_best_value': None,
            'diff_second': None,
            'pct_better_second': None,
            'avg_rest': None,
            'diff_avg': None,
            'pct_better_avg': None,
            'default_val': None,
            'diff_default': None,
            'pct_better_default': None
        }

        # Check for default bias row
        default_rows = df[df[label_col].astype(str).str.contains("defaultbias", case=False, na=False)]
        if not default_rows.empty:
            default_val = default_rows.iloc[0][col]
            diff_default = max_val - default_val
            
            result_entry['default_val'] = default_val
            result_entry['diff_default'] = diff_default
            
            if default_val != 0:
                result_entry['pct_better_default'] = (diff_default / default_val) * 100
                print(f"    Better than default bias ({default_val:.4f}) by: {diff_default:.4f} ({result_entry['pct_better_default']:.1f}%)")
            else:
                print(f"    Better than default bias ({default_val:.4f}) by: {diff_default:.4f}")
        
        if not rest.empty:
            second_best = rest.max()
            diff_second = max_val - second_best
            avg_rest = rest.mean()
            diff_avg = max_val - avg_rest
            
            result_entry['second_best_value'] = second_best
            result_entry['diff_second'] = diff_second
            result_entry['avg_rest'] = avg_rest
            result_entry['diff_avg'] = diff_avg
            
            if second_best != 0:
                result_entry['pct_better_second'] = (diff_second/second_best)*100
                print(f"    Better than 2nd best by: {diff_second:.4f} ({result_entry['pct_better_second']:.1f}%)")
            else:
                 print(f"    Better than 2nd best by: {diff_second:.4f}")

            if avg_rest != 0:
                result_entry['pct_better_avg'] = (diff_avg/avg_rest)*100
                print(f"    Better than average by:  {diff_avg:.4f} ({result_entry['pct_better_avg']:.1f}%)")
            else:
                print(f"    Better than average by:  {diff_avg:.4f}")
        else:
            print("    (Only one unique value or only one row)")

        results_list.append(result_entry)

    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze summary_comparison.csv files.")
    parser.add_argument("--root", type=str, default=".", help="Root directory to search.")
    args = parser.parse_args()

    find_and_analyze_csvs(args.root)
