import pandas as pd
import os
import glob
import sys
from pathlib import Path

# Add the synapse_sampling directory to the Python path
sys.path.append(str(Path(__file__).parent))
from wk_convert import calculate_bbox_coordinates

def main():
    # Path to the data directory containing Excel files
    data_dir = Path("data/data")
    
    # Find all Excel files matching the pattern bbox*.xlsx
    excel_files = glob.glob(str(data_dir / "bbox*.xlsx"))
    
    # Initialize an empty list to store all data
    all_data = []
    
    # Process each Excel file
    for excel_file in excel_files:
        # Extract bbox number from filename
        bbox_num = int(Path(excel_file).stem.replace("bbox", ""))
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file)
            
            # Check if the required columns exist
            required_cols = ["central_coord_1", "central_coord_2", "central_coord_3"]
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Required columns not found in {excel_file}. Skipping.")
                continue
            
            # Process each row in the Excel file
            for idx, row in df.iterrows():
                rel_x = row["central_coord_1"]
                rel_y = row["central_coord_2"]
                rel_z = row["central_coord_3"]
                
                # Calculate absolute coordinates
                abs_x, abs_y, abs_z = calculate_bbox_coordinates(rel_x, rel_y, rel_z, bbox_num)
                
                # Create a dictionary for this data point
                data_point = {
                    "bboxnumber": bbox_num,
                    "abspos1": abs_x,
                    "abspos2": abs_y,
                    "abspos3": abs_z,
                    "relpos1": rel_x,
                    "relpos2": rel_y,
                    "relpos3": rel_z
                }
                
                # Use existing Var1 if it exists, otherwise use row index + 1
                if "Var1" in row:
                    data_point["Var1"] = row["Var1"]
                else:
                    data_point["Var1"] = idx + 1
                
                all_data.append(data_point)
                
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
    
    # Create a DataFrame from all collected data
    result_df = pd.DataFrame(all_data)
    
    # Ensure Var1 is the first column
    if "Var1" in result_df.columns:
        cols = ["Var1"] + [col for col in result_df.columns if col != "Var1"]
        result_df = result_df[cols]
    
    # Save the DataFrame to a CSV file
    output_path = Path("data/absolute_positions.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Conversion complete. Saved {len(result_df)} positions to {output_path}")
    print(f"Data sample:\n{result_df.head()}")

if __name__ == "__main__":
    main() 