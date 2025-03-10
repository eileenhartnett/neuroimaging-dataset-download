# Neuroimaging Dataset Handler

A Python toolkit for working with neuroimaging datasets from OpenNeuro, designed to streamline the process of downloading, exploring, and analyzing brain imaging data.

## Overview

This project provides a framework for handling neuroimaging datasets that follow the BIDS (Brain Imaging Data Structure) format. It creates specialized handlers for different types of brain studies while managing the complexities of DataLad-based repositories.

## Key Features

- **Dataset-specific classes** for different study types (Flanker Task, Visual Working Memory, Word Recognition)
- **Robust download system** that handles DataLad symlinks and provides multiple fallback methods
- **Automatic file discovery** across different dataset organizations
- **Event extraction** to parse experimental timing information into structured tables
- **Brain visualization tools** showing multiple slice orientations

## Requirements

- Python 3.6+
- DataLad
- nibabel
- pandas
- matplotlib

## Quick Start

```python
# Create a dataset instance
from neuroimaging_datasets import create_dataset
dataset = create_dataset("ds000102")  # Flanker Task dataset

# Download the dataset
dataset.download_dataset()

# Find brain scan files
recordings = dataset.get_recording_filenames(modality="func")

# Extract experimental events
events_df = dataset.create_events_dataframe()
```

## Interactive Exploration

Run the interactive script to explore datasets with a user-friendly interface:

```
python test_neuroimaging_interactive.py
```

## Supported Datasets

- **Flanker Task** (ds000102): Attention and cognitive control
- **Visual Working Memory** (ds001771): Memory experiments
- **Word Recognition** (ds003097): Language processing

The system also provides basic support for any other OpenNeuro dataset through a generic implementation.

## Project Structure

- `neuroimaging_datasets.py`: Core implementation with dataset classes
- `test_neuroimaging_interactive.py`: Interactive exploration script

## Acknowledgments

- OpenNeuro for providing access to neuroimaging datasets
- DataLad for version-controlled data management
- BIDS community for standardized neuroimaging data organization