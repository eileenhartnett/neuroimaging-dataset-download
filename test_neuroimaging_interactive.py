# test_neuroimaging_interactive.py
from neuroimaging_dataset import NeuroimagingDataset

def test_dataset_interactive():
    """Test the NeuroimagingDataset class with user-selected dataset."""
    
    # Prompt user for dataset ID
    print("Available OpenNeuro datasets examples:")
    print("  ds000102 - Flanker task (recommended, small)")
    print("  ds001771 - Visual Working Memory (small)")
    print("  ds003097 - Word Recognition (medium)")
    print("  ds003592 - Visual Imagery (small)")
    print()
    
    dataset_id = input("Enter an OpenNeuro dataset ID (e.g., ds000102): ").strip()
    
    if not dataset_id:
        dataset_id = "ds000102"  # Default if empty
        print(f"Using default dataset: {dataset_id}")
    
    # Initialize dataset
    print(f"\nInitializing dataset: {dataset_id}")
    dataset = NeuroimagingDataset(dataset_id)
    
    # Download dataset
    print("\nDownloading dataset (this may take some time)...")
    success = dataset.download_dataset()
    print(f"Dataset download successful: {success}")
    
    if not success:
        print("Dataset download failed. Exiting.")
        return
    
    # Get dataset info
    print("\nGetting dataset information...")
    info = dataset.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Ask user which modality to explore
    print("\nCommon modalities in neuroimaging datasets:")
    print("  anat - Anatomical scans (T1w, T2w)")
    print("  func - Functional scans (fMRI)")
    print("  dwi  - Diffusion weighted imaging")
    modality = input("\nEnter modality to explore (default: anat): ").strip().lower()
    
    if not modality:
        modality = "anat"
    
    # Get recording filenames
    print(f"\nFinding {modality} recordings...")
    try:
        recordings = dataset.get_recording_filenames(modality=modality)
        print(f"Found {len(recordings)} {modality} recordings")
        
        # Print the first few recordings
        if recordings:
            print("\nSample recordings:")
            for recording in recordings[:5]:  # Show up to 5 examples
                print(f"  - {recording.relative_to(dataset.dataset_path)}")
            
            # Offer to load a specific recording
            load_file = input("\nWould you like to load one of these files? (y/n): ").strip().lower()
            
            if load_file == 'y':
                if len(recordings) > 0:
                    print("Loading the first file in the list...")
                    relative_path = str(recordings[0].relative_to(dataset.dataset_path))
                    print(f"\nLoading recording: {relative_path}")
                    try:
                        img, data = dataset.load_nifti(relative_path)
                        print(f"Data shape: {data.shape}")
                        print(f"Data type: {data.dtype}")
                        
                        # Optional: If matplotlib is available, visualize a slice
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(10, 8))
                            middle_slice = data.shape[2] // 2
                            plt.imshow(data[:, :, middle_slice], cmap='gray')
                            plt.title(f"Middle slice of {relative_path}")
                            plt.colorbar()
                            plt.axis('off')
                            plt.savefig("brain_slice.png")
                            print("\nSaved brain slice visualization to 'brain_slice.png'")
                            plt.close()
                        except Exception as viz_error:
                            print(f"\nMatplotlib visualization not available: {viz_error}")
                            
                    except Exception as e:
                        print(f"Error loading file: {e}")
                else:
                    print("No recordings found to load.")
        else:
            print(f"No {modality} recordings found in this dataset.")
    except Exception as e:
        print(f"Error getting recordings: {e}")
    
    # Create events dataframe
    print("\nChecking for events files...")
    try:
        events_df = dataset.create_events_dataframe()
        
        if not events_df.empty:
            print(f"Events dataframe shape: {events_df.shape}")
            print("\nSample events:")
            print(events_df.head())
            
            # Count events by task if available
            if 'task' in events_df.columns:
                print("\nEvents by task:")
                print(events_df['task'].value_counts())
        else:
            print("No events found in dataset.")
    except Exception as e:
        print(f"Error creating events dataframe: {e}")
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    test_dataset_interactive()
