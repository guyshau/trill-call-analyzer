# Trill Call Analyzer

This repository contains two Python scripts developed for analyzing `trill` vocalizations in common marmosets, based on the MarmAudio dataset (Lamothe et al., 2025). The analysis focuses on detecting temporal patterns and potential social coordination in vocal behavior.

## 📁 Files

- `trill_call_sequences.py`  
  Extracts sequences of vocalizations from the dataset and computes transition probabilities between call types.

- `trill_call_dynamics.py`  
  Analyzes the temporal distribution of trill calls over time (hour of day, day of week, etc.), and generates histograms and heatmaps.

## 📊 Features

- Sequence extraction based on `parent_name` (recording sessions)
- Weighted transition graph (using `networkx`)
- Time-based analysis and visualizations (using `matplotlib`, `seaborn`)
- Support for large datasets (`pandas`, `numpy`)

## 📦 Requirements

See `requirements.txt` below. All scripts were developed in Python 3.8+.

## 💡 Usage

To run the analysis:

```bash
python trill_call_sequences.py
python trill_call_dynamics.py
```

## 📚 Dataset

This project is based on the **MarmAudio** dataset:

**Lamothe, C., Obliger‑Debouche, M., Best, P., Trapeau, R., Ravel, S., Artières, T., Marxer, R., & Belin, P. (2025).**  
*MarmAudio: A large annotated dataset of vocalizations by common marmosets (Version 6).*  
Zenodo: https://doi.org/10.5281/zenodo.15017207  

## 🤝 Acknowledgements

This code builds upon the data structure and metadata conventions introduced in the MarmAudio dataset.  
Please cite their work if using or extending this project.

Developed as part of the final project for the course:  
**Animal Cognition (6170)**  
The Hebrew University of Jerusalem, June 2025.
