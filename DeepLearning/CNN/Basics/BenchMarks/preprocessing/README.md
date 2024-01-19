# Deep Learning Benchmark Data Generation README

This repository contains a script for generating and enhancing datasets for a deep learning benchmark using traffic sign images from the German Traffic Sign Recognition Benchmark (GTSRB). The README provides an overview of the project, its structure, and steps to generate and enhance datasets.

## Project Overview

- **Objective:** Generate and enhance datasets for deep learning benchmark using traffic sign images.

- **Datasets:** The original datasets are obtained from the GTSRB and are stored in CSV files (`Train.csv`, `Test.csv`, `Meta.csv`). Images are organized into subsets: Train, Test, and Meta.

- **Enhancement Modes:** The script supports various image enhancement modes, including RGB, RGB with Histogram Equalization (RGB-HE), Grayscale (L), Grayscale with Histogram Equalization (L-HE), Grayscale with Local Histogram Equalization (L-LHE), and Grayscale with Contrast Limited Adaptive Histogram Equalization (L-CLAHE).

- **Output:** The enhanced datasets are saved in HDF5 format in the specified output directory (`/archive/data`).

## Project Structure

- `archive/data`: Directory containing the enhanced datasets in HDF5 format.
- `main_script.py`: Main script for generating and enhancing datasets.
- `README.md`: Project documentation.
- Other necessary Python scripts and dependencies.

## Instructions

1. **Configuration:**
   - Set the `scale` and `output_dir` variables at the beginning of the script to control the size of the generated dataset and specify the output directory.

2. **Run the Script:**
   - Execute the `main_script.py` to generate and enhance datasets. Adjust the script parameters as needed.

```bash
python main_script.py
```

3. **Enhancement Modes:**
   - Modify the script to experiment with different enhancement modes. The script currently supports RGB, RGB-HE, L, L-HE, L-LHE, and L-CLAHE.

4. **Output:**
   - Enhanced datasets will be saved in the specified output directory in HDF5 format.

## Requirements

- NumPy
- Matplotlib
- Pandas
- scikit-image
- Pillow (PIL)

## Notes

- The script uses the GTSRB dataset in CSV format. Ensure that the CSV files (`Train.csv`, `Test.csv`, `Meta.csv`) are available in the specified paths.

Feel free to modify and experiment with the code to suit your specific use case or dataset. Happy data generation!