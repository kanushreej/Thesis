# Keyword Collection 

This directory contains scripts and data management strategies for collecting, labeling, and analyzing keywords related to specific political or research topics. The aim is to determine keyword relevance through moderator consensus and calculate agreement using the Adjusted Rand Index (ARI).

## Directory Structure

- `/ARI/Aggregated`:
  Contains CSV files with aggregated keywords for each issue.
- `/ARI/Labelled`:
  Stores CSV files with keywords labeled by individual moderators.
- `/Final`:
  Contains the final list of keywords agreed upon by the majority of moderators.
- `/Scripts`:
  Contains all Python scripts used for processing and analysis.

## Scripts

### 1. `aggregator.py`

Aggregates keywords from multiple methods and ensures deduplication.

- **Input**: CSV files from different keyword extraction methods.
- **Output**: Aggregated CSV files in the `/ARI/Aggregated` folder.

### 2. `set_relevance.py`

Enables moderators to label the relevance of keywords individually.

- **Input**: Aggregated keyword file.
- **Output**: Labeled CSV files in `/ARI/Labelled`.

### 3. `calculate_index.py`

Calculates the Adjusted Rand Index (ARI) to assess the agreement between moderators.

- **Input**: Labeled CSV files from multiple moderators.
- **Output**: Console output of ARI matrix.

### 4. `modify_relevance.py`

Allows moderators to modify their previous relevance ratings during a consensus meeting.

- **Input**: Labeled CSV file for a specific moderator.
- **Output**: Updated labeled CSV file.

### 5. `finalize.py`

Compiles the final list of keywords based on a majority rule from the consensus meeting.

- **Input**: Labeled CSV files from all moderators.
- **Output**: Final CSV files in /Final.

## Usage

```bash
python calculator.py --issue HealthcareUK
python set_relevance.py --moderator moderator_name --issue HealthcareUK
python calculate_index.py --issue HealthcareUK
python modify_relevance.py --moderator moderator_name --issue HealthcareUK
python finalize.py --issue HealthcareUK
