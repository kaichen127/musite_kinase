# Kinase-Substrate Inference

This container runs a pretrained kinase–substrate model.  
**Input**: FASTA files for substrate(s) and kinase(s) OR single substrate-kinase pair passed in through the command line.
**Output**: TXT (tab-separated) with a per-residue binary mask and positive residue indices.

## Requirements
- Download model_state.pth into this directory
- Docker installed and running

## Build

docker build -t kinase-model .

## Command Line Flags

### Required (choose one input mode)
- `--substrate_fasta` : Path to FASTA file of substrate sequences  
- `--kinase_fasta`    : Path to FASTA file of kinase sequences  
  *(FASTA mode: one-to-one pairing, both files must have the same number of sequences)*  

**OR**  

- `--substrate_seq`   : Single substrate sequence (string)  
- `--kinase_seq`      : Single kinase sequence (string)  
  *(Single-pair mode: both sequences must be provided)*  

---

### Optional
- `--weights`   : Path to model weights (`.pth`).  
  Default = `/app/model_state.pth` (baked into the Docker image)  
- `--device`    : Compute device: `auto`, `cpu`, or `cuda`.  
  Default = `auto` (use GPU if available, otherwise CPU)  
- `--threshold` : Classification cutoff for positive prediction.  
  Default = `0.5`  
- `--out_txt`   : Output TXT file path.  
  Default = `preds.txt`  
- `--batched`   : Enable batched inference (only if FASTA input is given).  
- `--batch_size`: Batch size when `--batched` is enabled.  
  Default = `4`  

## Example usage

### 1. Single Pair Input
Run with a single substrate and kinase sequence provided via flags:

docker run --rm kinase-model --substrate_seq "MDLPVGPGAAGPSNVPAF..." --kinase_seq "MSDVTIVKEGWVQKRGEYI..."

### 2. FASTA Input
Run with substrate and kinase FASTA files (must contain the same number of sequences):

docker run --rm -v $PWD:/data kinase-model --substrate_fasta /data/substrate.fasta --kinase_fasta /data/kinase.fasta

### 3. FASTA Input with Batched Inference
Enable batching for faster inference (on large files):

docker run --rm -v $PWD:/data kinase-model --substrate_fasta /data/substrate.fasta --kinase_fasta /data/kinase.fasta --batched --batch_size 4
