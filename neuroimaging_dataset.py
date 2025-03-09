# neuroimaging_dataset.py
import datalad.api as dl
from pathlib import Path
import nibabel as nib
import pandas as pd
import os
import glob

class NeuroimagingDataset:
    """Class for handling neuroimaging datasets from OpenNeuro."""
    
    def __init__(self, dataset_id, download_path=None):
        """
        Initialize a neuroimaging dataset.
        
        Parameters:
        -----------
        dataset_id : str
            OpenNeuro dataset ID (e.g., 'ds000102')
        download_path : Path or str, optional
            Path where datasets should be stored
        """
        self.dataset_id = dataset_id
        
        # Set download path (using provided path or creating a default)
        if download_path is None:
            self.download_path = Path(os.getcwd()) / "datasets"
        else:
            self.download_path = Path(download_path)
            
        # Set dataset path and initialize DataLad dataset
        self.dataset_path = self.download_path / dataset_id
        self.dataset = dl.Dataset(str(self.dataset_path))
        
    def download_dataset(self):
        """
        Ensure the dataset is downloaded, using alternative methods if needed.
        
        Returns:
        --------
        bool
            True if download successful or already installed
        """
        if not self.dataset.is_installed():
            print(f"Dataset {self.dataset_id} not installed. Installing...")
            try:
                # First attempt - standard DataLad install
                self.dataset.install(source=f"https://github.com/OpenNeuroDatasets/{self.dataset_id}.git")
                print(f"Dataset {self.dataset_id} installed successfully.")
                return True
            except Exception as e:
                print(f"Standard install failed: {e}")
                print("Trying alternative download method...")
                
                # Alternative approach - create directory and clone
                import subprocess
                import os
                
                try:
                    # Make sure the target directory exists
                    os.makedirs(self.dataset_path, exist_ok=True)
                    
                    # Use git directly
                    cmd = f"git clone https://github.com/OpenNeuroDatasets/{self.dataset_id}.git {self.dataset_path}"
                    subprocess.run(cmd, shell=True, check=True)
                    
                    # Initialize as DataLad dataset
                    self.dataset = dl.Dataset(str(self.dataset_path))
                    
                    print(f"Dataset {self.dataset_id} installed successfully with alternative method.")
                    return True
                except Exception as e2:
                    print(f"Alternative install also failed: {e2}")
                    return False
        else:
            print(f"Dataset {self.dataset_id} already installed.")
            return True
            
    def get_file(self, relative_path):
        """
        Ensure a specific file is downloaded.
        
        Parameters:
        -----------
        relative_path : str
            Path to the file, relative to the dataset root
            
        Returns:
        --------
        Path
            Full path to the downloaded file
        """
        full_path = self.dataset_path / relative_path
        
        if not full_path.exists() or full_path.is_symlink():
            print(f"File {full_path} is missing or a symlink. Downloading...")
            self.dataset.get(path=str(full_path))
            
        return full_path
    
    def load_nifti(self, relative_path):
        """
        Load a NIfTI file from the dataset.
        
        Parameters:
        -----------
        relative_path : str
            Path to the NIfTI file, relative to the dataset root
            
        Returns:
        --------
        tuple
            (nibabel image object, data array)
        """
        # Ensure file is downloaded
        full_path = self.get_file(relative_path)
        
        # Load the NIfTI file
        img = nib.load(str(full_path))
        data = img.get_fdata()
        print(f"Loaded {relative_path} successfully!")
        
        return img, data
    
    def get_recording_filenames(self, subject=None, modality=None, session=None, task=None):
        """
        Get filenames of all neuroimaging recordings in the dataset.
        
        Parameters:
        -----------
        subject : str or list, optional
            Filter by subject ID(s)
        modality : str, optional
            Filter by modality (e.g., 'anat', 'func', 'dwi')
        session : str, optional
            Filter by session ID
        task : str, optional
            Filter by task name
            
        Returns:
        --------
        list
            List of file paths matching the criteria
        """
        # Ensure dataset is downloaded
        if not self.dataset.is_installed():
            self.download_dataset()
        
        # Build search pattern
        pattern_parts = []
        
        # Add subject filter
        if subject:
            if isinstance(subject, list):
                subject_pattern = f"sub-({('|').join(subject)})"
                pattern_parts.append(subject_pattern)
            else:
                pattern_parts.append(f"sub-{subject}")
        else:
            pattern_parts.append("sub-*")
        
        # Add session filter
        if session:
            pattern_parts.append(f"ses-{session}")
        else:
            pattern_parts.append("*")
            
        # Add modality filter
        if modality:
            pattern_parts.append(modality)
        else:
            pattern_parts.append("*")
            
        # Add extra path level for standard BIDS organization
        pattern_parts.append("**")
        
        # Build pattern for specific file types
        search_pattern = str(self.dataset_path / "/".join(pattern_parts) / "*.nii*")
        
        # Find all matching files
        files = [Path(f) for f in glob.glob(search_pattern, recursive=True)]
        
        # Filter by task if specified
        if task and files:
            files = [f for f in files if f"task-{task}" in f.name]
            
        return files
    
    def create_events_dataframe(self, subject=None, task=None):
        """
        Create a DataFrame of events from the recordings.
        
        Parameters:
        -----------
        subject : str or list, optional
            Filter by subject ID(s)
        task : str, optional
            Filter by task name
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing events from all matching recordings
        """
        # Ensure dataset is downloaded
        if not self.dataset.is_installed():
            self.download_dataset()
            
        # Find all event files (typically named *_events.tsv in BIDS format)
        event_pattern = str(self.dataset_path / "**" / "*_events.tsv")
        event_files = glob.glob(event_pattern, recursive=True)
        
        all_events = []
        
        for event_file in event_files:
            file_path = Path(event_file)
            
            # Extract subject and task from filename
            file_name = file_path.name
            file_subject = None
            file_task = None
            
            # Parse BIDS filename components
            for part in file_name.split('_'):
                if part.startswith('sub-'):
                    file_subject = part
                elif part.startswith('task-'):
                    file_task = part.split('-')[1]
            
            # Apply filters
            if subject:
                if isinstance(subject, list):
                    if not any(f"sub-{s}" == file_subject for s in subject):
                        continue
                elif f"sub-{subject}" != file_subject:
                    continue
                    
            if task and file_task != task:
                continue
                
            try:
                # Read the events file
                events = pd.read_csv(file_path, sep='\t')
                
                # Add metadata columns
                events['subject'] = file_subject
                if file_task:
                    events['task'] = file_task
                events['source_file'] = str(file_path)
                
                # Get relative onset times
                if 'onset' in events.columns:
                    events['onset_time_sec'] = events['onset']
                
                all_events.append(events)
                
            except Exception as e:
                print(f"Error reading events file {file_path}: {e}")
                
        if not all_events:
            print("No event files found matching the criteria.")
            return pd.DataFrame()
            
        # Combine all events into a single DataFrame
        combined_events = pd.concat(all_events, ignore_index=True)
        
        return combined_events
        
    def get_dataset_info(self):
        """
        Get general information about the dataset.
        
        Returns:
        --------
        dict
            Dictionary containing dataset information
        """
        # Ensure dataset is downloaded
        if not self.dataset.is_installed():
            self.download_dataset()
            
        # Check for dataset_description.json
        desc_file = self.dataset_path / "dataset_description.json"
        
        info = {
            'dataset_id': self.dataset_id,
            'path': str(self.dataset_path)
        }
        
        if desc_file.exists():
            try:
                import json
                with open(desc_file, 'r') as f:
                    description = json.load(f)
                info.update(description)
            except Exception as e:
                print(f"Error reading dataset description: {e}")
                
        # Get subject count
        subjects = glob.glob(str(self.dataset_path / "sub-*"))
        info['subject_count'] = len(subjects)
        
        return info
