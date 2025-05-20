# Protein Contact DTW Analysis

This project compares wild-type and mutant protein contact maps using normalized DTW distances and unsupervised clustering.

## Project Structure

- `src/`: Core logic for preprocessing, DTW, clustering, plotting
- `scripts/`: Pipeline script to run the entire workflow
- `data/`: Contact map CSVs (chain A and B)
- `labels/`: Mapping of trajectories to groups
- `output/`: Results (plots, distance matrix)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
