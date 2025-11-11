# Kinase-Substrate Inference (Python)

This script runs a pretrained kinase–substrate model.  
**Input**: CSV file containing substrate and kinase protein sequences.
**Output**: CSV file containing per-residue binary predictions and positive residue indices.

## Requirements
- Download model_state.pth into this directory
- Python ≥ 3.9
- Install dependencies via ```pip install -r requirements.txt ```

## Usage
- Add your own data file to the `data` directory or use `example_data.csv` for an example dataset.
- Modify the configurations in `configs/config.yaml` accordingly. Default settings have been set, but you will need to modify the input data path if using your own dataset.
- Run inference directly with ```bash python inference.py ```.