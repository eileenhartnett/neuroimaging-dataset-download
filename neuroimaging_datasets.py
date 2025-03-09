# neuroimaging_datasets.py
#
# This file implements a class hierarchy for working with neuroimaging datasets from OpenNeuro.
# It provides a base abstract class and specific implementations for different datasets.
# The design follows object-oriented principles with inheritance and polymorphism.
#
# Key components:
# - BaseNeuroimagingDataset: Abstract base class with common functionality
# - Dataset-specific classes (FlankerTask, VisualWorkingMemory, etc.)
# - Factory function to create appropriate dataset instances

import datalad.api as dl        # DataLad API for dataset management
from pathlib import Path        # Object-oriented filesystem paths
import nibabel as nib           # Library for neuroimaging file formats (NIfTI)
import pandas as pd             # Data manipulation library
import os                       # Operating system utilities
import glob                     # File pattern matching
import json                     # JSON parsing for dataset metadata
import requests                 # HTTP requests for metadata retrieval
from abc import ABC, abstractmethod  # Abstract base class functionality
import numpy as np              # Numerical operations

class BaseNeuroimagingDataset(ABC):
    """
    Abstract base class for neuroimaging datasets.
    
    This class provides common functionality for all datasets while requiring
    dataset-specific implementations through abstract methods. It handles
    dataset downloading, file management, and data extraction in a way that
    can be customized for each dataset's structure.
    """
    
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
        # Store the dataset ID for reference
        self.dataset_id = dataset_id
        
        # Try to get a friendly name for the dataset from OpenNeuro API
        # This makes displays more user-friendly than just showing dataset IDs
        self.dataset_name = self._get_dataset_name()
        
        # Set the download path - either use provided path or create default in current directory
        if download_path is None:
            # Create 'datasets' directory in current working directory if no path specified
            self.download_path = Path(os.getcwd()) / "datasets"
        else:
            # Convert provided path to Path object for consistent handling
            self.download_path = Path(download_path)
            
        # Full path to this specific dataset (download_path/dataset_id)
        self.dataset_path = self.download_path / dataset_id
        
        # Initialize a DataLad dataset object (does NOT download data yet)
        self.dataset = dl.Dataset(str(self.dataset_path))
        
        # Initialize dataset-specific configurations
        # These will be overridden by subclasses to customize behavior
        self.expected_modalities = ['anat', 'func', 'dwi']  # Standard BIDS modalities
        self.primary_tasks = []  # Will be filled by subclasses
        self.metadata = {}  # Will hold dataset description and other metadata
        
    def _get_dataset_name(self):
        """
        Get the friendly name of the dataset from OpenNeuro API if possible.
        Falls back to dataset ID if API call fails.
        
        Returns:
        --------
        str
            Human-readable dataset name or the dataset ID if name not available
        """
        try:
            # Try to fetch dataset information from OpenNeuro API
            response = requests.get(f"https://openneuro.org/crn/datasets/{self.dataset_id}", timeout=5)
            if response.status_code == 200:
                # If successful, extract the dataset name
                data = response.json()
                return data.get('name', self.dataset_id)
        except Exception:
            # If any error occurs (network, timeout, parsing), ignore and use default
            pass
        # Default to using the dataset ID as name if API call fails
        return self.dataset_id

    def download_dataset(self):
        """
        Download the dataset, handling different formats and potential errors.
        
        This method implements multiple download strategies:
        1. Standard DataLad download (preferred)
        2. Git clone (alternative if DataLad fails)
        3. Direct AWS S3 download (last resort)
        
        Returns:
        --------
        bool
            True if download successful or already installed
        """
        # First check if dataset is already installed to avoid redundant downloads
        if not self.dataset.is_installed():
            print(f"Dataset {self.dataset_id} ({self.dataset_name}) not installed. Installing...")
            
            # Track download attempts for diagnostic purposes
            download_attempts = []
            
            # METHOD 1: Try standard DataLad download
            # This is the preferred method as it handles versioning and metadata
            try:
                # Connect to the OpenNeuro GitHub repository for this dataset
                self.dataset.install(source=f"https://github.com/OpenNeuroDatasets/{self.dataset_id}.git")
                print(f"Dataset {self.dataset_id} installed successfully via DataLad.")
                # Call post-download setup for dataset-specific initialization
                self._post_download_setup()
                return True
            except Exception as e:
                # If DataLad install fails, record the error and try next method
                download_attempts.append(f"DataLad install failed: {str(e)}")
                
            # METHOD 2: Try Git clone as an alternative
            # This is more direct and might work when DataLad has issues
            try:
                import subprocess
                # Create target directory if it doesn't exist
                os.makedirs(self.dataset_path, exist_ok=True)
                
                # Use subprocess to execute a git clone command
                cmd = f"git clone https://github.com/OpenNeuroDatasets/{self.dataset_id}.git {self.dataset_path}"
                subprocess.run(cmd, shell=True, check=True)
                
                # Re-initialize as DataLad dataset to maintain consistent interface
                self.dataset = dl.Dataset(str(self.dataset_path))
                print(f"Dataset {self.dataset_id} installed successfully via Git.")
                # Call post-download setup for dataset-specific initialization
                self._post_download_setup()
                return True
            except Exception as e:
                # If Git clone fails, record the error and try next method
                download_attempts.append(f"Git clone failed: {str(e)}")
            
            # METHOD 3: Try direct AWS S3 download as last resort
            # OpenNeuro hosts datasets on AWS S3, so we can access them directly
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.client import Config
                
                print("Attempting S3 download...")
                # Create S3 client without authentication (public bucket)
                s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                # Create target directory
                os.makedirs(self.dataset_path, exist_ok=True)
                
                # Download dataset_description.json to get metadata
                # This is a small file that contains basic dataset information
                s3.download_file(
                    'openneuro', 
                    f"{self.dataset_id}/dataset_description.json",
                    str(self.dataset_path / "dataset_description.json")
                )
                
                # Note: In a complete implementation, we would download all files
                # or implement on-demand downloading, but this is simplified
                print(f"Dataset {self.dataset_id} metadata retrieved via S3. Full download would continue here.")
                # Call post-download setup for dataset-specific initialization
                self._post_download_setup()
                return True
            except Exception as e:
                # If S3 download fails, record the error
                download_attempts.append(f"S3 download failed: {str(e)}")
            
            # If all methods failed, display detailed error information
            print("All download methods failed. Details:")
            for attempt in download_attempts:
                print(f"- {attempt}")
            return False
        else:
            # Dataset already installed, just set up metadata
            print(f"Dataset {self.dataset_id} already installed.")
            self._post_download_setup()
            return True
    
    def _post_download_setup(self):
        """
        Perform any dataset-specific setup after download.
        
        This method is called after successful dataset installation,
        and is intended to be overridden by subclasses to add custom
        initialization that depends on dataset contents.
        
        Base implementation loads metadata from dataset_description.json.
        """
        # Look for dataset_description.json (standard BIDS metadata file)
        desc_file = self.dataset_path / "dataset_description.json"
        if desc_file.exists():
            try:
                # Parse JSON metadata if file exists
                with open(desc_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error reading dataset description: {e}")
    
    def get_file(self, relative_path):
        """
        Ensure a specific file is downloaded.
        
        DataLad uses a lazy loading approach where files are initially
        represented as symlinks. This method resolves the symlink by
        downloading the actual file content when needed.
        
        Parameters:
        -----------
        relative_path : str
            Path to the file, relative to the dataset root
            
        Returns:
        --------
        Path
            Full path to the downloaded file
        """
        # Construct full path by joining dataset path and relative path
        full_path = self.dataset_path / relative_path
        
        # Check if file needs to be downloaded (doesn't exist or is a symlink)
        if not full_path.exists() or full_path.is_symlink():
            print(f"File {full_path} is missing or a symlink. Downloading...")
            try:
                # First try standard DataLad get method to resolve symlink
                self.dataset.get(path=str(full_path))
            except Exception as e:
                # If DataLad fails, try alternative download method
                print(f"DataLad get failed: {e}. Trying alternative method...")
                self._get_file_alternative(relative_path)
            
        # Return the full path to the now-downloaded file
        return full_path
    
    def _get_file_alternative(self, relative_path):
        """
        Alternative method to get file if DataLad fails.
        
        This is a placeholder for subclasses to implement specific
        alternative download methods tailored to each dataset's
        storage structure.
        
        Parameters:
        -----------
        relative_path : str
            Path to the file, relative to the dataset root
        """
        pass  # Default implementation does nothing
    
    def load_nifti(self, relative_path):
        """
        Load a NIfTI file from the dataset.
        
        NIfTI (Neuroimaging Informatics Technology Initiative) is a standard
        file format for storing neuroimaging data.
        
        Parameters:
        -----------
        relative_path : str
            Path to the NIfTI file, relative to the dataset root
            
        Returns:
        --------
        tuple
            (nibabel image object, data array)
            - image object contains metadata and header information
            - data array is the actual brain imaging data as a numpy array
        """
        # First ensure the file is downloaded using get_file method
        full_path = self.get_file(relative_path)
        
        # Load the NIfTI file using nibabel
        img = nib.load(str(full_path))
        # Extract the actual data as a numpy array
        data = img.get_fdata()
        print(f"Loaded {relative_path} successfully!")
        
        # Return both the image object (with header/metadata) and data array
        return img, data
    
    def get_recording_filenames(self, subject=None, modality=None, session=None, task=None):
        """
        Get filenames of all neuroimaging recordings in the dataset.
        
        This method searches for recording files that match the specified
        criteria, adapting to different dataset structures and BIDS conventions.
        
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
        # Ensure dataset is downloaded before searching
        if not self.dataset.is_installed():
            success = self.download_dataset()
            if not success:
                return []  # Return empty list if download failed
        
        # Check if requested modality is valid for this dataset
        if modality and modality not in self.expected_modalities:
            # Warn but continue if modality doesn't match expected values
            print(f"Warning: Modality '{modality}' is not standard for this dataset.")
            print(f"Expected modalities: {', '.join(self.expected_modalities)}")
        
        # Build the search pattern based on BIDS directory structure
        pattern_parts = []
        
        # Step 1: Add subject filter
        if subject:
            if isinstance(subject, list):
                # For multiple subjects, create regex pattern with alternatives
                subject_pattern = f"sub-({('|').join(subject)})"
                pattern_parts.append(subject_pattern)
            else:
                # For single subject, just use the ID
                pattern_parts.append(f"sub-{subject}")
        else:
            # If no subject specified, match all subjects with wildcard
            pattern_parts.append("sub-*")
        
        # Step 2: Add session filter
        if session:
            # Include specific session ID
            pattern_parts.append(f"ses-{session}")
        else:
            # If no session specified, match any directory level
            pattern_parts.append("*")
            
        # Step 3: Add modality filter
        if modality:
            # Filter by specific modality (anat, func, dwi)
            pattern_parts.append(modality)
        else:
            # If no modality specified, match any directory level
            pattern_parts.append("*")
            
        # Step 4: Add recursive wildcard to match files at any deeper level
        pattern_parts.append("**")
        
        # Step 5: Build full search pattern for NIfTI files (.nii or .nii.gz)
        search_pattern = str(self.dataset_path / "/".join(pattern_parts) / "*.nii*")
        
        # Step 6: Find all matching files using glob with recursive search
        files = [Path(f) for f in glob.glob(search_pattern, recursive=True)]
        
        # Step 7: Filter by task if specified
        # Task is often encoded in the filename rather than directory structure
        if task and files:
            files = [f for f in files if f"task-{task}" in f.name]
            
        # Step 8: Apply dataset-specific processing to organize files
        files = self._process_recording_files(files)
        
        return files
    
    def _process_recording_files(self, files):
        """
        Apply dataset-specific processing to the found recording files.
        
        This method allows subclasses to implement custom organization
        or filtering of files based on dataset-specific conventions.
        
        Parameters:
        -----------
        files : list
            List of file paths found by get_recording_filenames
            
        Returns:
        --------
        list
            Processed list of file paths
        """
        # Default implementation returns files unchanged
        # Subclasses can override to implement custom processing
        return files
    
    def create_events_dataframe(self, subject=None, task=None):
        """
        Create a DataFrame of events from the recordings.
        
        In neuroimaging experiments, 'events' represent stimuli, responses,
        or other occurrences during a scan. This method extracts event
        information from TSV files and organizes them in a DataFrame.
        
        Parameters:
        -----------
        subject : str or list, optional
            Filter by subject ID(s)
        task : str, optional
            Filter by task name
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing events from all matching recordings with columns:
            - Standard BIDS columns (onset, duration, trial_type, etc.)
            - Additional columns for subject, task, and source file
            - Dataset-specific derived columns added by _process_events
        """
        # Ensure dataset is downloaded before searching for events
        if not self.dataset.is_installed():
            success = self.download_dataset()
            if not success:
                return pd.DataFrame()  # Return empty DataFrame if download failed
            
        # Step 1: Find all event files (named *_events.tsv in BIDS format)
        # These files contain trial information with timing
        event_pattern = str(self.dataset_path / "**" / "*_events.tsv")
        event_files = glob.glob(event_pattern, recursive=True)
        
        # List to store individual event DataFrames before concatenation
        all_events = []
        
        # Step 2: Process each event file
        for event_file in event_files:
            file_path = Path(event_file)
            
            # Step 3: Extract subject and task from filename using BIDS conventions
            # BIDS filename format: sub-<subject>_task-<task>_run-<run>_events.tsv
            file_name = file_path.name
            file_subject = None
            file_task = None
            
            # Parse filename components (separated by underscores)
            for part in file_name.split('_'):
                if part.startswith('sub-'):
                    file_subject = part
                elif part.startswith('task-'):
                    file_task = part.split('-')[1]
            
            # Step 4: Apply subject filter if specified
            if subject:
                if isinstance(subject, list):
                    # For multiple subjects, check if file's subject is in the list
                    if not any(f"sub-{s}" == file_subject for s in subject):
                        continue  # Skip this file if subject doesn't match
                elif f"sub-{subject}" != file_subject:
                    continue  # Skip this file if subject doesn't match
                    
            # Step 5: Apply task filter if specified
            if task and file_task != task:
                continue  # Skip this file if task doesn't match
                
            try:
                # Step 6: Read the events file as a DataFrame (tab-separated values)
                events = pd.read_csv(file_path, sep='\t')
                
                # Step 7: Add metadata columns to identify the source of each event
                events['subject'] = file_subject
                if file_task:
                    events['task'] = file_task
                events['source_file'] = str(file_path)
                
                # Step 8: Standardize onset time column naming
                # BIDS uses 'onset' column for event timing in seconds
                if 'onset' in events.columns:
                    events['onset_time_sec'] = events['onset']
                
                # Step 9: Apply dataset-specific event processing
                # This allows subclasses to add derived columns specific to each task
                events = self._process_events(events, file_task)
                
                # Add this event DataFrame to our collection
                all_events.append(events)
                
            except Exception as e:
                # Log errors but continue processing other files
                print(f"Error reading events file {file_path}: {e}")
                
        # Step 10: Check if any events were found
        if not all_events:
            print("No event files found matching the criteria.")
            return pd.DataFrame()  # Return empty DataFrame if no events found
            
        # Step 11: Combine all individual event DataFrames into one
        combined_events = pd.concat(all_events, ignore_index=True)
        
        return combined_events
    
    def _process_events(self, events, task):
        """
        Process events DataFrame with dataset-specific logic.
        
        This method allows subclasses to add dataset-specific columns
        or transformations to the events DataFrame, such as:
        - Adding derived measures (e.g., response correctness)
        - Categorizing trials
        - Computing performance metrics
        
        Parameters:
        -----------
        events : pandas.DataFrame
            DataFrame containing events for a single recording
        task : str or None
            Task name for these events
            
        Returns:
        --------
        pandas.DataFrame
            Processed events DataFrame
        """
        # Default implementation returns events unchanged
        # Subclasses should override to add task-specific processing
        return events
        
    def get_dataset_info(self):
        """
        Get general information about the dataset.
        
        Collects metadata from dataset_description.json, counts subjects,
        and adds dataset-specific information from subclasses.
        
        Returns:
        --------
        dict
            Dictionary containing dataset information including:
            - dataset_id: OpenNeuro identifier
            - dataset_name: Human-readable name
            - path: Local filesystem path
            - BIDS metadata from dataset_description.json
            - subject_count: Number of subjects
            - Custom information added by subclasses
        """
        # Ensure dataset is downloaded first
        if not self.dataset.is_installed():
            success = self.download_dataset()
            if not success:
                # Return minimal info if download failed
                return {'dataset_id': self.dataset_id, 'download_status': 'failed'}
            
        # Step 1: Start with basic info dictionary
        info = {
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'path': str(self.dataset_path)
        }
        
        # Step 2: Add metadata from dataset_description.json
        # This typically includes dataset name, authors, references, etc.
        info.update(self.metadata)
                
        # Step 3: Count subjects by finding all subject directories
        # In BIDS, subjects are in directories named 'sub-<id>'
        subjects = glob.glob(str(self.dataset_path / "sub-*"))
        info['subject_count'] = len(subjects)
        
        # Step 4: Add dataset-specific information from subclass
        info.update(self._get_custom_dataset_info())
        
        return info
    
    def _get_custom_dataset_info(self):
        """
        Get dataset-specific information.
        
        This method allows subclasses to add custom metadata specific
        to each dataset type, such as task paradigms, cognitive domains,
        or experimental design details.
        
        Returns:
        --------
        dict
            Dictionary with dataset-specific information
        """
        # Default implementation returns empty dict
        # Subclasses should override to add custom information
        return {}
    
    @abstractmethod
    def get_recommended_analyses(self):
        """
        Return recommended analyses for this specific dataset.
        
        This abstract method forces subclasses to implement
        dataset-specific analysis recommendations based on
        the nature of the experimental paradigm.
        
        Returns:
        --------
        list
            List of dictionaries with recommended analyses, each containing:
            - name: Analysis name
            - description: What the analysis examines
            - type: Analysis type (contrast, parametric, correlation)
            - Additional type-specific fields (conditions, parameters, etc.)
        """
        # Abstract method must be implemented by subclasses
        pass


# Specific Dataset Classes

class FlankerTaskDataset(BaseNeuroimagingDataset):
    """
    Class for the Flanker Task dataset (ds000102).
    
    The Flanker task investigates cognitive control and inhibition.
    Participants respond to a central stimulus while ignoring flanking stimuli
    that are either congruent (same direction) or incongruent (opposite direction).
    """
    
    def __init__(self, download_path=None):
        """
        Initialize Flanker Task dataset.
        
        Parameters:
        -----------
        download_path : Path or str, optional
            Path where dataset should be stored
        """
        # Call parent constructor with fixed dataset ID
        super().__init__("ds000102", download_path)
        
        # Set dataset-specific configuration
        # This dataset only contains anatomical and functional scans
        self.expected_modalities = ['anat', 'func']
        # The only task in this dataset is the flanker task
        self.primary_tasks = ['flanker']
    
    def _post_download_setup(self):
        """
        Custom setup after download for Flanker dataset.
        
        Adds task-specific metadata about the experimental conditions
        and key data columns to aid in analysis.
        """
        # First call parent implementation to load basic metadata
        super()._post_download_setup()
        
        # Add detailed information about the flanker task
        self.task_info = {
            'flanker': {
                'description': 'Flanker task with congruent and incongruent trials',
                'conditions': ['congruent', 'incongruent'],
                'key_columns': ['trial_type', 'response_time']
            }
        }
    
    def _process_events(self, events, task):
        """
        Process events specific to the Flanker task.
        
        Adds derived columns useful for Flanker task analysis:
        - correctness of responses
        - identification of slow responses
        
        Parameters:
        -----------
        events : pandas.DataFrame
            Events for a single recording
        task : str or None
            Task name
            
        Returns:
        --------
        pandas.DataFrame
            Processed events with additional columns
        """
        # Only process if this is flanker task data
        if task == 'flanker':
            # Add specific columns for flanker task analysis
            if 'trial_type' in events.columns and 'response_time' in events.columns:
                # Calculate if response was correct based on trial type and response
                # In flanker tasks, correct response depends on congruency condition
                if 'stim' in events.columns and 'resp' in events.columns:
                    events['correct'] = (
                        # For congruent trials, response should match stimulus
                        (events['trial_type'] == 'congruent') & (events['resp'] == events['stim']) |
                        # For incongruent trials, response should NOT match stimulus
                        (events['trial_type'] == 'incongruent') & (events['resp'] != events['stim'])
                    )
                
                # Flag slow responses (> 2 standard deviations from mean)
                # This is useful for identifying outlier responses
                if 'response_time' in events.columns:
                    mean_rt = events['response_time'].mean()
                    std_rt = events['response_time'].std()
                    events['slow_response'] = events['response_time'] > (mean_rt + 2*std_rt)
        
        return events
    
    def get_recommended_analyses(self):
        """
        Return recommended analyses for the Flanker Task dataset.
        
        Suggests standard analyses typically performed on flanker task data,
        focusing on cognitive control and inhibition measures.
        
        Returns:
        --------
        list
            List of recommended analyses for this dataset
        """
        return [
            {
                'name': 'Congruent vs Incongruent Contrast',
                'description': 'Compare brain activation between congruent and incongruent trials',
                'type': 'contrast',
                'conditions': ['congruent', 'incongruent']
            },
            {
                'name': 'Response Time Parametric Analysis',
                'description': 'Analyze brain regions with activity correlated with response time',
                'type': 'parametric',
                'parameter': 'response_time'
            }
        ]
    
    def _get_custom_dataset_info(self):
        """
        Add custom information specific to Flanker dataset.
        
        Provides experimental context and cognitive domains relevant
        to this particular dataset.
        
        Returns:
        --------
        dict
            Dictionary with Flanker-specific metadata
        """
        return {
            'task_paradigm': 'Flanker',
            'cognitive_domains': ['attention', 'cognitive control', 'inhibition'],
            'tasks': self.primary_tasks
        }


class VisualWorkingMemoryDataset(BaseNeuroimagingDataset):
    """
    Class for the Visual Working Memory dataset (ds001771).
    
    This dataset examines working memory using a visual task with varying
    memory load levels (1, 2, 4, and 6 items).
    """
    
    def __init__(self, download_path=None):
        """
        Initialize Visual Working Memory dataset.
        
        Parameters:
        -----------
        download_path : Path or str, optional
            Path where dataset should be stored
        """
        # Call parent constructor with fixed dataset ID
        super().__init__("ds001771", download_path)
        
        # Set dataset-specific configuration
        # This dataset contains anatomical, functional, and field map scans
        self.expected_modalities = ['anat', 'func', 'fmap']
        # The only task in this dataset is the working memory task
        self.primary_tasks = ['workingmemory']
    
    def _post_download_setup(self):
        """
        Custom setup after download for Visual Working Memory dataset.
        
        Adds task-specific metadata about the experimental conditions
        and memory load levels used in the experiment.
        """
        # First call parent implementation to load basic metadata
        super()._post_download_setup()
        
        # Add detailed information about the working memory task
        self.task_info = {
            'workingmemory': {
                'description': 'Visual working memory task with various load levels',
                'conditions': ['load1', 'load2', 'load4', 'load6'],
                'key_columns': ['trial_type', 'accuracy', 'response_time']
            }
        }
    
    def _process_events(self, events, task):
        """
        Process events specific to the Visual Working Memory task.
        
        Adds derived columns useful for working memory analysis:
        - Categorical load level labels
        - Memory capacity estimates
        
        Parameters:
        -----------
        events : pandas.DataFrame
            Events for a single recording
        task : str or None
            Task name
            
        Returns:
        --------
        pandas.DataFrame
            Processed events with additional columns
        """
        # Only process if this is working memory task data
        if task == 'workingmemory':
            # Add columns specific to working memory analysis
            if 'load' in events.columns:
                # Convert load to categorical for easier grouping and visualization
                events['load_level'] = 'load' + events['load'].astype(str)
                
                # Calculate memory capacity if accuracy is available
                # K = N * (H - FA) where N is load, H is hit rate, FA is false alarm rate
                if 'accuracy' in events.columns:
                    # This is a simplified calculation and would need proper hit/false alarm data
                    # in a real implementation. Here we just use accuracy as a proxy.
                    events['memory_capacity'] = events['load'] * events['accuracy']
        
        return events
    
    def get_recommended_analyses(self):
        """
        Return recommended analyses for the Visual Working Memory dataset.
        
        Suggests analyses specific to working memory paradigms,
        particularly those examining capacity limits and load effects.
        
        Returns:
        --------
        list
            List of recommended analyses for this dataset
        """
        return [
            {
                'name': 'Load-Dependent Activity',
                'description': 'Identify regions showing increased activity with memory load',
                'type': 'parametric',
                'parameter': 'load'
            },
            {
                'name': 'High vs Low Load Contrast',
                'description': 'Compare brain activation between high and low memory load trials',
                'type': 'contrast',
                'conditions': ['load6', 'load1']
            },
            {
                'name': 'Memory Capacity Correlation',
                'description': 'Correlate brain activity with individual memory capacity estimates',
                'type': 'correlation',
                'measure': 'memory_capacity'
            }
        ]
    
    def _get_custom_dataset_info(self):
        """
        Add custom information specific to Visual Working Memory dataset.
        
        Provides experimental context and details about the memory
        load manipulation central to this experiment.
        
        Returns:
        --------
        dict
            Dictionary with working memory specific metadata
        """
        return {
            'task_paradigm': 'Visual Working Memory',
            'cognitive_domains': ['working memory', 'attention', 'visual processing'],
            'memory_load_levels': [1, 2, 4, 6],
            'tasks': self.primary_tasks
        }


class WordRecognitionDataset(BaseNeuroimagingDataset):
    """
    Class for the Word Recognition dataset (ds003097).
    
    This dataset examines neural responses during word recognition,
    focusing on language processing and reading.
    """
    
    def __init__(self, download_path=None):
        """
        Initialize Word Recognition dataset.
        
        Parameters:
        -----------
        download_path : Path or str, optional
            Path where dataset should be stored
        """
        # Call parent constructor with fixed dataset ID
        super().__init__("ds003097", download_path)
        
        # Set dataset-specific configuration
        # This dataset contains anatomical and functional scans
        self.expected_modalities = ['anat', 'func']
        # The only task is word recognition
        self.primary_tasks = ['wordrecognition']
    
    def _process_recording_files(self, files):
        """
        Apply additional organization specific to Word Recognition dataset.
        
        For multi-run experiments, organizes files by run number to
        maintain temporal order of recordings.
        
        Parameters:
        -----------
        files : list
            List of file paths
            
        Returns:
        --------
        list
            Reorganized list of file paths
        """
        # Organize files by run (for multi-run experiments)
        files_by_run = {}
        for file in files:
            run_match = None
            # Extract run number from filename (run-01, run-02, etc.)
            for part in file.name.split('_'):
                if part.startswith('run-'):
                    run_match = part
                    break
            
            # Group files by run number
            if run_match:
                if run_match not in files_by_run:
                    files_by_run[run_match] = []
                files_by_run[run_match].append(file)
        
        # If we successfully organized by run, sort by run number
        if files_by_run:
            # Sort files by run number and flatten the list
            ordered_files = []
            for run in sorted(files_by_run.keys()):
                ordered_files.extend(files_by_run[run])
            return ordered_files
        
        # If no run numbers found, return original list
        return files
    
    def _process_events(self, events, task):
        """
        Process events specific to the Word Recognition task.
        
        Adds linguistic properties and categorizations for words:
        - Word length
        - Complexity categories
        
        Parameters:
        -----------
        events : pandas.DataFrame
            Events for a single recording
        task : str or None
            Task name
            
        Returns:
        --------
        pandas.DataFrame
            Processed events with additional columns
        """
        # Only process if this is word recognition task data
        if task == 'wordrecognition':
            # Add word-specific processing
            if 'word' in events.columns:
                # Calculate word length (number of characters)
                events['word_length'] = events['word'].str.len()
                
                # Simple categorization based on word length
                # In a real implementation, this would use more sophisticated
                # linguistic metrics like frequency, concreteness, etc.
                events['word_complexity'] = pd.cut(
                    events['word_length'],
                    bins=[0, 4, 7, 20],  # Bin edges: 0-4 chars, 5-7 chars, 8+ chars
                    labels=['simple', 'medium', 'complex']  # Category labels
                )
        
        return events
    
    def get_recommended_analyses(self):
        """
        Return recommended analyses for the Word Recognition dataset.
        
        Suggests analyses specific to language processing,
        focusing on word complexity and recognition processes.
        
        Returns:
        --------
        list
            List of recommended analyses for this dataset
        """
        return [
            {
                'name': 'Word Complexity Effect',
                'description': 'Analyze brain regions showing differential response to word complexity',
                'type': 'parametric',
                'parameter': 'word_length'
            },
            {
                'name': 'Recognition vs Baseline',
                'description': 'Compare brain activation during word recognition vs baseline',
                'type': 'contrast',
                'conditions': ['word', 'baseline']
            }
        ]
    
    def _get_custom_dataset_info(self):
        """
        Add custom information specific to Word Recognition dataset.
        
        Provides context about the linguistic aspects and
        cognitive domains relevant to this dataset.
        
        Returns:
        --------
        dict
            Dictionary with word recognition specific metadata
        """
        return {
            'task_paradigm': 'Word Recognition',
            'cognitive_domains': ['language', 'reading', 'memory'],
            'tasks': self.primary_tasks,
            'linguistic_features': ['word_length', 'word_complexity']
        }


# Dataset Factory for easy creation of appropriate dataset class

def create_dataset(dataset_id, download_path=None):
    """
    Factory function to create the appropriate dataset class based on dataset ID.
    
    This function determines which specific dataset class to instantiate
    based on the dataset ID, or falls back to a generic implementation
    for unknown datasets.
    
    Parameters:
    -----------
    dataset_id : str
        OpenNeuro dataset ID
    download_path : Path or str, optional
        Path where datasets should be stored
    
    Returns:
    --------
    BaseNeuroimagingDataset
        An instance of the appropriate dataset class
    """
    # Map dataset IDs to their specific class implementations
    dataset_classes = {
        'ds000102': FlankerTaskDataset,
        'ds001771': VisualWorkingMemoryDataset,
        'ds003097': WordRecognitionDataset
    }
    
    # If we have a specific class for this dataset, use it
    if dataset_id in dataset_classes:
        return dataset_classes[dataset_id](download_path)
    
    # For unknown datasets, create a generic implementation
    # We can't return BaseNeuroimagingDataset directly as it's abstract
    class GenericDataset(BaseNeuroimagingDataset):
        """Generic dataset implementation for unknown dataset IDs."""
        
        def get_recommended_analyses(self):
            """Provide basic analysis recommendations for unknown datasets."""
            return [
                {
                    'name': 'Basic Activation Analysis',
                    'description': 'Standard analysis of task vs baseline activation',
                    'type': 'basic'
                }
            ]
    
    # Return an instance of the generic dataset class
    return GenericDataset(dataset_id, download_path)