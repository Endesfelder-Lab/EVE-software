import os
import sys

"""
This script renames files in folders to their folder name and a sequential number.
Mainly used because files acquired with Metavision are named with the date and time of acquisition,
which is not very convenient for analysis.
"""

def rename_files_in_folders(target_directory):
    # check if path exists
    if not os.path.exists(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
        return

    # Walk through the directory structure
    for root, dirs, files in os.walk(target_directory):
        # Skip the root folder itself, we only want to process subfolders
        if root == target_directory:
            continue
            
        # Get the name of the current folder
        folder_name = os.path.basename(root)
        
        # Sort files to ensure sequential clips (C001, C002) stay in order
        files.sort()
        
        # specific counter for this folder
        counter = 1
        
        for filename in files:
            # Get the file extension (e.g., .mp4, .MOV, .wav)
            name, ext = os.path.splitext(filename)
            
            # Skip system files like .DS_Store on Mac
            if filename.startswith('.'):
                continue

            # Create the new name
            # Logic: FolderName_01.ext, FolderName_02.ext
            new_filename = f"{folder_name}_{counter:02d}{ext}"
            
            # Get full paths
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            
            # Rename
            try:
                os.rename(old_path, new_path)
                print(f"[OK] Renamed: {filename} -> {new_filename}")
                counter += 1
            except OSError as e:
                print(f"[ERROR] Could not rename {filename}: {e}")

if __name__ == "__main__":
    print("--- File Renamer Script ---")
    # You can paste the path directly when asked
    path_input = input("Enter the full path to your main directory: ").strip()
    
    # Remove quotes if the user pasted path as "C:\Path"
    path_input = path_input.replace('"', '').replace("'", "")
    
    rename_files_in_folders(path_input)
    print("\nProcessing complete.")