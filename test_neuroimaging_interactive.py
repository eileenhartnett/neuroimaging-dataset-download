# Add auto-download function
def auto_download_file(file_path, dataset, verbose=True):
    """
    Automatically download a file if it's a symlink and auto-download is enabled.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file to download
    dataset : BaseNeuroimagingDataset
        Dataset instance with auto_download setting
    verbose : bool, optional
        Whether to print status messages
        
    Returns:
    --------
    bool
        True if file is available (was already downloaded or was successfully downloaded),
        False if file remains a symlink
    """
    import os
    import subprocess
    from pathlib import Path
    
    # Convert to Path object for consistent handling
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    # Check if file exists and is a symlink
    if not os.path.exists(file_path):
        if verbose:
            print(f"File does not exist: {file_path}")
        return False
    
    if not os.path.islink(file_path):
        # File already downloaded
        return True
    
    # If auto-download is disabled, just return status
    if not getattr(dataset, 'auto_download', False):
        if verbose:
            print(f"File is a symlink but auto-download is disabled: {file_path}")
        return False
    
    # Auto-download is enabled, try to download the file
    if verbose:
        print(f"Auto-downloading file: {file_path}")
    
    # Calculate relative path from dataset root
    try:
        relative_path = file_path.relative_to(dataset.dataset_path)
    except ValueError:
        # If file is not within dataset path, use full path
        relative_path = file_path
    
    # Method 1: Try dataset's get_file method
    try:
        dataset.get_file(str(relative_path))
        if not os.path.islink(file_path):
            if verbose:
                print("✓ File downloaded successfully via dataset.get_file()")
            return True
    except Exception as e:
        if verbose:
            print(f"Could not download via dataset.get_file(): {e}")
    
    # Method 2: Try direct datalad command
    try:
        result = subprocess.run(
            ["datalad", "get", str(file_path)],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and not os.path.islink(file_path):
            if verbose:
                print("✓ File downloaded successfully via datalad get")
            return True
        elif verbose:
            print(f"Datalad get failed with code {result.returncode}")
    except Exception as e:
        if verbose:
            print(f"Error running datalad command: {e}")
    
    # Method 3: Try git-annex directly
    try:
        result = subprocess.run(
            ["git", "annex", "get", str(file_path)],
            cwd=str(dataset.dataset_path),
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and not os.path.islink(file_path):
            if verbose:
                print("✓ File downloaded successfully via git-annex get")
            return True
        elif verbose:
            print(f"Git annex get failed with code {result.returncode}")
    except Exception as e:
        if verbose:
            print(f"Error running git-annex command: {e}")
    
    # All methods failed
    if verbose:
        print(f"❌ Could not download file: {file_path}")
    return False# Add helper function for Y/N questions
def ask_yes_no(prompt, default=None):
    """
    Ask a yes/no question and return a boolean.
    
    Parameters:
    -----------
    prompt : str
        The question to ask
    default : bool, optional
        Default answer if user just presses Enter
        
    Returns:
    --------
    bool
        True for yes, False for no
    """
    # Determine default value display
    if default is True:
        prompt += " [Y/n]"
    elif default is False:
        prompt += " [y/N]"
    else:
        prompt += " [y/n]"
    
    # Keep asking until valid input
    while True:
        answer = input(prompt).strip().lower()
        
        # Handle empty input with default
        if not answer and default is not None:
            return default
        
        # Process valid inputs
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'.")# test_neuroimaging_interactive.py
#
# This script provides an interactive interface for working with neuroimaging datasets.
# It demonstrates the usage of the neuroimaging_datasets module by allowing users to:
# 1. Select and download datasets
# 2. Explore recording files
# 3. Examine events data
# 4. View analysis recommendations
# 5. Visualize brain images
#
# The script is designed to be user-friendly with menu-driven navigation and
# clear explanations of available options.

# Import the dataset classes and factory function from our module
from neuroimaging_datasets import create_dataset, FlankerTaskDataset, VisualWorkingMemoryDataset, WordRecognitionDataset

# Import libraries for visualization and data handling
import matplotlib.pyplot as plt  # For creating brain visualizations
import numpy as np              # For numerical operations
import pandas as pd             # For data manipulation
import os                       # For file operations

def test_dataset_interactive():
    """
    Main function that provides an interactive interface to test neuroimaging datasets.
    
    This function guides the user through:
    1. Selecting a dataset
    2. Downloading the dataset
    3. Displaying dataset information
    4. Providing a menu of exploration options
    """
    
    # Display welcome header and available datasets
    print("\n===== NEUROIMAGING DATASET EXPLORER =====\n")
    print("Available OpenNeuro datasets:")
    print("  1. Flanker Task (ds000102) - Attention and cognitive control")
    print("  2. Visual Working Memory (ds001771) - Memory load experiment")
    print("  3. Word Recognition (ds003097) - Language processing")
    print("  4. Custom dataset ID - Enter your own OpenNeuro dataset ID")
    print()
    
    # Get user selection for which dataset to explore
    choice = input("Select a dataset (1-4): ").strip()
    
    # Process the user's selection to determine dataset ID and class
    dataset_id = None
    if choice == '1':
        # Flanker Task dataset
        dataset_id = 'ds000102'
        dataset_class = FlankerTaskDataset
    elif choice == '2':
        # Visual Working Memory dataset
        dataset_id = 'ds001771'
        dataset_class = VisualWorkingMemoryDataset
    elif choice == '3':
        # Word Recognition dataset
        dataset_id = 'ds003097'
        dataset_class = WordRecognitionDataset
    elif choice == '4':
        # Custom dataset - let user enter specific ID
        dataset_id = input("Enter an OpenNeuro dataset ID (e.g., ds000102): ").strip()
        dataset_class = None  # Will use factory to determine class
    else:
        # Handle invalid input by using default dataset
        print("Invalid selection. Using default dataset (Flanker Task).")
        dataset_id = 'ds000102'
        dataset_class = FlankerTaskDataset
    
    # Create the appropriate dataset using the factory function
    # The factory automatically selects the right class based on dataset_id
    print(f"\nInitializing dataset: {dataset_id}")
    dataset = create_dataset(dataset_id)
    
    # Download the selected dataset
    # This may take some time depending on dataset size and internet connection
    print("\nDownloading dataset (this may take some time)...")
    success = dataset.download_dataset()
    print(f"Dataset download successful: {success}")
    
    # Exit if download failed
    if not success:
        print("Dataset download failed. Exiting.")
        return
    
    # Configure auto-download setting
    auto_download = ask_yes_no("\nWould you like to enable automatic file downloading? (y/n): ", default=True)
    
    if auto_download:
        print("\nAutomatic file downloading is ENABLED.")
        print("The system will attempt to download actual file content when needed.")
        print("This may slow down operations but ensures files are available when needed.")
    else:
        print("\nAutomatic file downloading is DISABLED.")
        print("You will need to manually download files with 'datalad get' commands.")
    
    # Store auto-download setting
    dataset.auto_download = auto_download
    
    # Display dataset information in a nicely formatted table
    print("\n===== DATASET INFORMATION =====")
    info = dataset.get_dataset_info()
    
    # Calculate maximum key width for aligned display
    max_key_width = max(len(key) for key in info.keys())
    
    # Display each piece of dataset information
    for key, value in info.items():
        if isinstance(value, (list, dict)):
            # Format complex values (lists and dicts) for better display
            if isinstance(value, list):
                # Join list items with commas
                value_str = ", ".join(str(item) for item in value)
            else:  # dict
                # Format each key-value pair in the dictionary
                value_str = "\n" + "\n".join(f"  {k}: {v}" for k, v in value.items())
            print(f"{key.ljust(max_key_width)} : {value_str}")
        else:
            # Simple values displayed directly
            print(f"{key.ljust(max_key_width)} : {value}")
    
    # Main interaction loop - present options until user chooses to exit
    while True:
        # Display menu of exploration options
        print("\n===== EXPLORATION OPTIONS =====")
        print("1. View recording files")
        print("2. Examine events data")
        print("3. View recommended analyses")
        print("4. Load and visualize an image")
        print("5. Auto-download status:", "ENABLED" if dataset.auto_download else "DISABLED")
        print("6. Exit")
        
        # Get user's menu selection
        option = input("\nSelect an option (1-6): ").strip()
        
        # Process the selected option
        if option == '1':
            # Option 1: Browse recording files
            explore_recordings(dataset)
        elif option == '2':
            # Option 2: Examine experimental events
            explore_events(dataset)
        elif option == '3':
            # Option 3: View analysis recommendations
            show_recommended_analyses(dataset)
        elif option == '4':
            # Option 4: Load and visualize a brain image
            load_and_visualize(dataset)
        elif option == '5':
            # Option 5: Toggle auto-download setting
            dataset.auto_download = not dataset.auto_download
            status = "ENABLED" if dataset.auto_download else "DISABLED"
            print(f"\nAutomatic file downloading is now {status}")
        elif option == '6':
            # Option 6: Exit the program
            print("\nExiting dataset explorer. Goodbye!")
            break
        else:
            # Handle invalid menu selection
            print("Invalid option. Please try again.")


def explore_recordings(dataset):
    """
    Allow users to explore recording files in the dataset.
    
    This function helps users:
    1. Select which modality (anat, func, dwi) to explore
    2. Filter by subject or task if desired
    3. View a list of matching recording files
    4. Automatically download files if enabled
    
    Parameters:
    -----------
    dataset : BaseNeuroimagingDataset
        The dataset instance to explore
    """
    print("\n===== RECORDING FILES =====")
    
    # Import necessary modules
    import os
    import glob
    from pathlib import Path
    
    # Step 1: Ask user to select which modality to explore
    print("\nAvailable modalities:")
    for idx, modality in enumerate(dataset.expected_modalities, 1):
        print(f"  {idx}. {modality}")
    
    modality_choice = input("\nSelect a modality (number or name): ").strip()
    
    # Process modality selection - handle both numeric and text input
    modality = None
    try:
        # If input is a number, convert to corresponding modality
        idx = int(modality_choice) - 1
        if 0 <= idx < len(dataset.expected_modalities):
            modality = dataset.expected_modalities[idx]
    except ValueError:
        # If not a number, check if input matches a valid modality name
        if modality_choice in dataset.expected_modalities:
            modality = modality_choice
    
    # If invalid input, use the first modality as default
    if not modality:
        print(f"Invalid modality. Using default: {dataset.expected_modalities[0]}")
        modality = dataset.expected_modalities[0]
    
    # Step 2: Ask if user wants to filter by specific subject
    subject = input("\nEnter subject ID to filter (leave empty for all): ").strip()
    
    # Step 3: For functional data, offer task filtering
    task = None
    if modality == 'func' and dataset.primary_tasks:
        # Display available tasks for this dataset
        print("\nAvailable tasks:")
        for idx, task_name in enumerate(dataset.primary_tasks, 1):
            print(f"  {idx}. {task_name}")
        
        # Get task selection from user
        task_choice = input("\nSelect a task (number or name, leave empty for all): ").strip()
        
        # Process task selection - handle both numeric and text input
        try:
            # If input is a number, convert to corresponding task
            idx = int(task_choice) - 1
            if 0 <= idx < len(dataset.primary_tasks):
                task = dataset.primary_tasks[idx]
        except ValueError:
            # If not a number, check if input matches a valid task name
            if task_choice in dataset.primary_tasks:
                task = task_choice
    
    # Step 4: Direct file search as a first approach - more reliable
    print(f"\nSearching for {modality} recordings...")
    
    # Create path pattern based on BIDS structure
    search_pattern = f"{dataset.dataset_path}/sub-*"
    if subject:
        search_pattern = f"{dataset.dataset_path}/sub-{subject}"
    
    # Add modality to search pattern
    search_pattern += f"/**/{modality}/**/*.nii*"
    
    # Find all matching files using glob
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter by task if specified
    if task and all_files:
        all_files = [f for f in all_files if f"task-{task}" in f]
    
    # Convert to Path objects
    all_files = [Path(f) for f in all_files]
    
    # If direct search found files, use those
    if all_files:
        recordings = all_files
        print(f"Found {len(recordings)} {modality} recordings through direct search.")
    else:
        # Fall back to dataset method if direct search finds nothing
        recordings = dataset.get_recording_filenames(subject=subject, modality=modality, task=task)
        if recordings:
            print(f"Found {len(recordings)} {modality} recordings using dataset method.")
    
    # Handle case where no matching files are found
    if not recordings:
        print(f"No {modality} recordings found with the specified criteria.")
        print("\nTroubleshooting:")
        print("1. Try downloading the files explicitly:")
        print(f"   cd {dataset.dataset_path}")
        print(f"   datalad get -J 4 sub-*/{modality}/*.nii*")
        print("2. Check if the dataset follows BIDS structure")
        print("3. Try a different modality")
        return
    
    # Ask if user wants to download some files
    auto_download_enabled = getattr(dataset, 'auto_download', False)
    symlinks_exist = any(os.path.islink(file) for file in recordings)
    
    if symlinks_exist:
        if auto_download_enabled:
            print("\nSome files are symlinks. Auto-download is enabled.")
            download_option = ask_yes_no("Download a sample file now? (y/n): ", default=True)
            
            if download_option and recordings:
                # Download one file as a sample
                sample_file = recordings[0]
                print(f"\nDownloading sample file: {sample_file.relative_to(dataset.dataset_path)}")
                success = auto_download_file(sample_file, dataset)
                if success:
                    print("✓ Sample file downloaded successfully!")
                else:
                    print("❌ Could not download sample file.")
        else:
            print("\nSome files are symlinks. Auto-download is disabled.")
            print("Use option 5 in the main menu to enable auto-download,")
            print("or manually download files with:")
            print(f"cd {dataset.dataset_path}")
            print(f"datalad get -J 4 sub-*/{modality}/*.nii*")
    
    # Step 5: Display the found recordings in a structured format
    print(f"\nFound {len(recordings)} matching recordings:")
    
    # Group recordings by subject for better organization
    recordings_by_subject = {}
    for recording in recordings:
        # Extract subject ID from file path
        subject = "unknown"
        path_str = str(recording)
        
        # Look for sub-XX pattern in path
        import re
        match = re.search(r'sub-([a-zA-Z0-9]+)', path_str)
        if match:
            subject = f"sub-{match.group(1)}"
        
        # Create subject groups
        if subject not in recordings_by_subject:
            recordings_by_subject[subject] = []
        recordings_by_subject[subject].append(recording)
    
    # Display recordings grouped by subject
    record_index = 1
    for subject, files in recordings_by_subject.items():
        # Print subject header
        print(f"\n  {subject}:")
        
        # Show up to 5 files per subject to avoid overwhelming output
        for file in files[:5]:
            # Show path relative to dataset root for clarity
            try:
                rel_path = file.relative_to(dataset.dataset_path)
                is_symlink = os.path.islink(file)
                symlink_status = " (symlink)" if is_symlink else ""
                print(f"    {record_index}. {rel_path}{symlink_status}")
            except ValueError:
                # If relative_to fails, show the full path
                print(f"    {record_index}. {file}")
            record_index += 1
        
        # Indicate if there are more files not shown
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more files")
    
    # Print summary of found files
    print(f"\nTotal: {len(recordings)} files across {len(recordings_by_subject)} subjects")
    print("\nNote: Files marked as 'symlink' need to be downloaded with 'datalad get' before use")


def explore_events(dataset):
    """
    Allow users to explore experimental events data in the dataset.
    
    This function helps users:
    1. Select a specific task to explore (if multiple are available)
    2. Filter events by subject if desired
    3. View event statistics and details
    4. Save events to CSV if desired
    
    Parameters:
    -----------
    dataset : BaseNeuroimagingDataset
        The dataset instance to explore
    """
    print("\n===== EVENTS DATA =====")
    
    # Step 1: Determine which task to examine
    task = None
    if dataset.primary_tasks:
        if len(dataset.primary_tasks) == 1:
            # If only one task is available, use it automatically
            task = dataset.primary_tasks[0]
            print(f"Using task: {task}")
        else:
            # If multiple tasks are available, let user choose
            print("\nAvailable tasks:")
            for idx, task_name in enumerate(dataset.primary_tasks, 1):
                print(f"  {idx}. {task_name}")
            
            # Get task selection from user
            task_choice = input("\nSelect a task (number or name, leave empty for all): ").strip()
            
            # Process task selection - handle both numeric and text input
            try:
                # If input is a number, convert to corresponding task
                idx = int(task_choice) - 1
                if 0 <= idx < len(dataset.primary_tasks):
                    task = dataset.primary_tasks[idx]
            except ValueError:
                # If not a number, check if input matches a valid task name
                if task_choice in dataset.primary_tasks:
                    task = task_choice
    
    # Step 2: Ask if user wants to filter by specific subject
    subject = input("\nEnter subject ID to filter (leave empty for all): ").strip()
    
    # Step 3: Create events dataframe with the specified filters
    print("\nLoading events data...")
    events_df = dataset.create_events_dataframe(subject=subject, task=task)
    
    # Handle case where no events are found
    if events_df.empty:
        print("No events found matching the specified criteria.")
        return
    
    # Step 4: Display basic statistics about the events
    print(f"\nFound {len(events_df)} events.")
    
    # Display dataframe structure (columns, types, etc.)
    print("\nEvents DataFrame Structure:")
    print(f"Shape: {events_df.shape} (rows × columns)")
    print(f"Columns: {list(events_df.columns)}")
    print("\nData types:")
    for col, dtype in events_df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Display the first few rows to show event examples
    print("\nSample events:")
    print(events_df.head())
    
    # Step 5: Display counts by categories if available
    # Find categorical columns with reasonable number of unique values
    categorical_columns = []
    for col in events_df.columns:
        # Include object type columns and specifically named columns
        if events_df[col].dtype == 'object' or col in ['trial_type', 'task', 'subject']:
            # Only include if there aren't too many unique values
            if events_df[col].nunique() < 20:  # Limit to reasonable number of categories
                categorical_columns.append(col)
    
    # Display value counts for categorical columns
    if categorical_columns:
        print("\nEvents by category:")
        for col in categorical_columns:
            counts = events_df[col].value_counts()
            print(f"\n  {col}:")
            for val, count in counts.items():
                print(f"    {val}: {count}")
    
    # Step 6: Offer to save events to CSV file for further analysis
    save_option = input("\nDo you want to save events to CSV? (y/n): ").strip().lower()
    if save_option == 'y':
        # Create descriptive filename based on dataset, task, and subject filters
        filename = f"events_{dataset.dataset_id}"
        if task:
            filename += f"_{task}"
        if subject:
            filename += f"_sub-{subject}"
        filename += ".csv"
        
        # Save to CSV and confirm
        events_df.to_csv(filename, index=False)
        print(f"Events saved to {filename}")


def show_recommended_analyses(dataset):
    """
    Display recommended analyses for the dataset.
    
    This function presents dataset-specific analysis recommendations
    provided by the dataset class, showing appropriate analyses based
    on the experimental paradigm.
    
    Parameters:
    -----------
    dataset : BaseNeuroimagingDataset
        The dataset instance to show recommendations for
    """
    print("\n===== RECOMMENDED ANALYSES =====")
    
    # Get recommended analyses for this specific dataset
    analyses = dataset.get_recommended_analyses()
    
    # Handle case where no recommendations are available
    if not analyses:
        print("No specific analyses recommended for this dataset.")
        return
    
    # Display each recommended analysis with details
    for i, analysis in enumerate(analyses, 1):
        # Print analysis name and type
        print(f"\n{i}. {analysis['name']}")
        print(f"   Type: {analysis['type']}")
        print(f"   Description: {analysis['description']}")
        
        # Show additional details based on analysis type
        if 'conditions' in analysis:
            # For contrast analyses, show conditions to compare
            print(f"   Conditions: {', '.join(analysis['conditions'])}")
        if 'parameter' in analysis:
            # For parametric analyses, show parameter to use
            print(f"   Parameter: {analysis['parameter']}")
        if 'measure' in analysis:
            # For correlation analyses, show measure to correlate with
            print(f"   Measure: {analysis['measure']}")


def load_and_visualize(dataset):
    """
    Load and visualize a brain image from the dataset.
    
    This function:
    1. Finds suitable images in the dataset
    2. EXPLICITLY downloads the actual file content if needed
    3. Loads a selected image
    4. Creates visualizations showing different slice orientations
    5. Saves the visualization to a file
    
    Parameters:
    -----------
    dataset : BaseNeuroimagingDataset
        The dataset instance to visualize
    """
    print("\n===== IMAGE VISUALIZATION =====")
    
    # Import necessary modules
    import os
    import subprocess
    import glob
    from pathlib import Path
    import nibabel as nib
    
    # Step 1: Find ANY NIfTI files in the dataset (using direct file search)
    print("\nSearching for NIfTI files...")
    all_nifti_files = glob.glob(str(dataset.dataset_path) + "/**/*.nii*", recursive=True)
    
    if not all_nifti_files:
        print("No NIfTI files found in the dataset. Nothing to visualize.")
        return
    
    print(f"Found {len(all_nifti_files)} NIfTI files in the dataset.")
    
    # Step 2: Select a file to download and visualize
    # For simplicity, choose the first one that's likely to be anatomical
    selected_file = None
    
    # First try to find an anatomical (T1w) file which is best for visualization
    for nifti_file in all_nifti_files:
        if 'anat' in nifti_file and ('T1w' in nifti_file or 'T1' in nifti_file):
            selected_file = nifti_file
            print(f"Selected anatomical T1w file: {os.path.basename(selected_file)}")
            break
    
    # If no T1w file found, just take the first NIfTI file
    if not selected_file and all_nifti_files:
        selected_file = all_nifti_files[0]
        print(f"Selected file: {os.path.basename(selected_file)}")
    
    if not selected_file:
        print("Failed to select a file for visualization.")
        return
    
    # Convert to Path object for easier handling
    selected_path = Path(selected_file)
    
    # Check if auto-download is enabled and download if needed
    auto_download_enabled = getattr(dataset, 'auto_download', False)
    is_symlink = os.path.islink(selected_file)
    
    if is_symlink:
        if auto_download_enabled:
            print("\nFile is a symlink. Auto-downloading actual content...")
            success = auto_download_file(selected_file, dataset)
            if not success:
                print("❌ Could not download file content. Visualization may fail.")
        else:
            print("\nFile is a symlink and auto-download is disabled.")
            print("Visualization may fail unless you first download the file content.")
            download_now = ask_yes_no("Would you like to download this file now? (y/n): ", default=True)
            if download_now:
                success = auto_download_file(selected_file, dataset)
                if not success:
                    print("❌ Could not download file content. Visualization may fail.")
    
    # Step 5: Load the NIfTI file directly with nibabel
    print("\nLoading image data directly with nibabel...")
    try:
        # Load directly with nibabel instead of using dataset method
        img = nib.load(selected_file)
        data = img.get_fdata()
        
        print(f"✓ Successfully loaded image!")
        print(f"Data shape: {data.shape}")  # Display dimensions of the volume
        print(f"Data type: {data.dtype}")   # Display data type
        
        # Step 6: Create visualization
        visualize_brain_image(data, os.path.basename(selected_file))
        
    except Exception as e:
        # Handle any errors during loading or visualization
        print(f"✗ Error loading or visualizing file: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run these commands in your terminal:")
        print(f"   cd {dataset.dataset_path}")
        print("   datalad get -J 4 sub-*/anat/*.nii*")
        print("   datalad get -J 4 sub-*/func/*.nii*")
        print("2. Make sure datalad and git-annex are installed:")
        print("   pip install datalad")
        print("   brew install git-annex (macOS) or apt-get install git-annex (Linux)")
        print("3. If all else fails, try downloading the dataset directly from OpenNeuro website")


def visualize_brain_image(data, title):
    """
    Create a comprehensive visualization of a brain image.
    
    This function generates a three-panel visualization showing
    slices in different orientations (sagittal, coronal, axial).
    
    Parameters:
    -----------
    data : numpy.ndarray
        3D or 4D brain imaging data
    title : str
        Title for the visualization, typically the file path
    """
    try:
        # Create visualization appropriate to the data dimensionality
        if len(data.shape) > 2:
            # For 3D volumes, create multi-slice views
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Get middle slices for each orientation
            x_mid = data.shape[0] // 2
            y_mid = data.shape[1] // 2
            z_mid = data.shape[2] // 2
            
            # Sagittal view (YZ plane) - slice through the middle of X dimension
            axes[0].imshow(data[x_mid, :, :].T, cmap='gray', origin='lower')
            axes[0].set_title('Sagittal View')
            axes[0].axis('off')  # Hide axis ticks for cleaner image
            
            # Coronal view (XZ plane) - slice through the middle of Y dimension
            axes[1].imshow(data[:, y_mid, :].T, cmap='gray', origin='lower')
            axes[1].set_title('Coronal View')
            axes[1].axis('off')
            
            # Axial view (XY plane) - slice through the middle of Z dimension
            # This is the standard "top-down" view in most medical imaging
            axes[2].imshow(data[:, :, z_mid], cmap='gray', origin='lower')
            axes[2].set_title('Axial View')
            axes[2].axis('off')
            
            # Add an overall title for the figure
            plt.suptitle(f"Brain Image: {title}")
            
        elif len(data.shape) == 2:
            # For 2D data (rare in neuroimaging), create a single panel
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='gray')
            plt.title(f"Image: {title}")
            plt.axis('off')
            plt.colorbar()  # Add intensity scale
        elif len(data.shape) == 4:
            # For 4D data (time series), show middle volume
            middle_vol = data.shape[3] // 2
            print(f"4D data detected - showing volume {middle_vol} of {data.shape[3]}")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Get middle slices for each orientation in the middle volume
            x_mid = data.shape[0] // 2
            y_mid = data.shape[1] // 2
            z_mid = data.shape[2] // 2
            
            # Sagittal view from middle volume
            axes[0].imshow(data[x_mid, :, :, middle_vol].T, cmap='gray', origin='lower')
            axes[0].set_title('Sagittal View')
            axes[0].axis('off')
            
            # Coronal view from middle volume
            axes[1].imshow(data[:, y_mid, :, middle_vol].T, cmap='gray', origin='lower')
            axes[1].set_title('Coronal View')
            axes[1].axis('off')
            
            # Axial view from middle volume
            axes[2].imshow(data[:, :, z_mid, middle_vol], cmap='gray', origin='lower')
            axes[2].set_title('Axial View')
            axes[2].axis('off')
            
            plt.suptitle(f"Brain Image: {title} (Volume {middle_vol})")
        
        # Create directory for saved visualizations
        os.makedirs('visualizations', exist_ok=True)
        
        # Create safe filename by replacing path separators
        safe_title = str(title).replace('/', '_').replace('\\', '_')
        filename = f"visualizations/brain_{safe_title}.png"
        
        # Save figure to file and close to free memory
        plt.savefig(filename)
        plt.close()
        
        print(f"\nBrain visualization saved to: {filename}")
        
    except Exception as e:
        # Handle any visualization errors
        print(f"Error creating visualization: {e}")
        # Print more debug info
        print(f"Data shape: {data.shape}, Data type: {data.dtype}")
        if isinstance(data, np.ndarray):
            print(f"Data range: {data.min()} to {data.max()}")
            print(f"Any NaNs: {np.isnan(data).any()}")
        import traceback
        traceback.print_exc()


# This is a standard Python idiom that allows the file to be:
# 1. Run directly as a standalone program, or
# 2. Imported as a module without running the test automatically
if __name__ == "__main__":
    # When run directly, start the interactive test
    test_dataset_interactive()