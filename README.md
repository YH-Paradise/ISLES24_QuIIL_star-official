# ISLES24_QUiiL_star

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation of our solution for the [ISLES 2024 Challenge (Ischemic Stroke Lesion Segmentation)](https://isles24.grand-challenge.org/). 

## 📌 Introduction
* **Challenge Link:** [ISLES 24 Official Website](https://isles24.grand-challenge.org/)
* **Paper:** [ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand?](https://arxiv.org/abs/2408.10966)
* **Key Features:**
  * Utilization of Hybrid-based model architecture (Mobile Vision Transformer + Convolutional Neural Network)
  * Efficient 3D volume data loading and augmentation using `Torchio` and `SimpleITK`.
  * [Key Contribution 1: e.g., Anatomical Edge-and-Gravity Informed Segmentation (AEGIS) framework applied to stroke lesions.]
  * [Key Contribution 2: e.g., Custom distance map generation for boundary-aware learning.]

---

## ⚙️ Requirements & Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/[repository-name].git
cd [repository-name]
```

2. Create a virtual environment and install the required packages:
```bash
conda create -n isles24 python=3.9 -y
conda activate isles24
pip install -r requirements.txt
```
> **Note:** Core dependencies include `torch`, `torchio`, `SimpleITK`, `nibabel`, `scikit-image`, and `pandas`.

---

## 📂 Dataset Preparation

Download the official ISLES24 dataset and organize it according to the following directory structure. We use the NIfTI (`.nii.gz`) format.

```text
data/
 ├── train/
 │   ├── case_001/
 │   │   ├── case_001_DWI.nii.gz
 │   │   ├── case_001_ADC.nii.gz
 │   │   └── case_001_seg.nii.gz (Ground Truth)
 │   └── ...
 └── test/
     ├── case_101/
     └── ...
```

Run the preprocessing script to handle resampling, skull-stripping (if applicable), and normalization:
```bash
python scripts/preprocess.py --data_dir ./data --out_dir ./data_preprocessed
```

---

## 🚀 Training

To train the model on the preprocessed dataset, configure your hyperparameters in `configs/base_config.yaml` and run:

```bash
python train.py --config configs/base_config.yaml --batch_size 2 --epochs 200
```
* Training logs and checkpoints will be automatically saved in the `checkpoints/` directory.

---

## 🧠 Inference (Evaluation)

To generate predictions on the test set using your trained weights:

```bash
python inference.py --data_dir ./data/test --checkpoint_path checkpoints/best_model.pth --out_dir ./predictions
```
The predicted segmentation masks will be saved as `.nii.gz` files in the specified `--out_dir`, ready to be zipped and submitted to the challenge platform.

---

## 📊 Results

Performance metrics on the internal validation set or the official ISLES24 leaderboard:

| Model               | Dice Score (↑) | HD95 (↓)  | ASSD (↓)  |
|:--------------------|:--------------:|:---------:|:---------:|
| Baseline (3D UNet)  |      0.xx      |   xx.xx   |   xx.xx   |
| **Ours (Proposed)** |    **0.xx**    | **xx.xx** | **xx.xx** |

*(Consider adding qualitative results here. Insert 1-2 screenshots or a GIF comparing the Ground Truth vs. Prediction using a NIfTI viewer.)*
![Result Sample](docs/images/sample_result.png)

---

## 📝 Citation

If you find this code or our methodology useful in your research, please consider citing our work:

```bibtex
@inproceedings{yang2024isles,
  title={Your Paper Title for the ISLES24 Challenge},
  author={Yang, Hyun and [Co-authors] and Kwak, Jin Tae},
  booktitle={ISLES 2024 Challenge Proceedings},
  year={2024}
}
```

---

## 🤝 Acknowledgements

* Special thanks to the ISLES24 organizers for providing the dataset and hosting the challenge.

## Instructions
1. Create directory "resources/weights"
2. Download pretrained weight from
    https://drive.google.com/file/d/1af6u3eBRlzoPA_Ycmdz8twgU_L8MS6Qd/view?usp=sharing
5. Add "final_weight_2.pt" to "resources/weights"

The structure of repository should look like this:
```bash
ISLES24_QUIIL_star/
├── Best_Model
├── file_dir_csvs/
│   └── ...
├── models/
│   ├── MoReT_3D/
│   │   └── mobilevit_v3_block.py
│   │   └── moret_3d.py
│   │   └── vit_block.py
│   └── model_structure.py
├── resources/
│   ├── weights/
│   │   └── final_weight_2.pt  # This will be downloaded through the link above.
├── utils/
│   ├── common/
│   │   └── ...
│   ├── sample_data/  # These are selected randomly for test.
│   │   └── train/ 
│   │   │   └── sub-stokre0003_ses-01_cta.nii.gz
│   │   │   └── sub-stroke0003_ses-01_ctp.nii.gz
│   │   │   └── sub-stroke0003_ses-01_ncct.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_cta.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_ctp.nii.gz
│   │   │   └── sub-stroke0003_ses-01_space-ncct_tmax.nii.gz
│   │   │   └── sub-stroke0003_ses-02_adc.nii.gz
│   │   │   └── sub-stroke0003_ses-02_dwi.nii.gz
│   │   │   └── sub-stroke0003_ses-02_lesion-msk.nii.gz
│   │   └── val/
│   │   │   └── sub-stroke0004_ses-01_ncct.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_cta.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_ctp.nii.gz
│   │   │   └── sub-stroke0004_ses-01_space-ncct_tmax.nii.gz
│   │   │   └── sub-stroke0004_ses-02_adc.nii.gz
│   │   │   └── sub-stroke0004_ses-02_dwi.nii.gz
│   │   │   └── sub-stroke0004_ses-02_lesion-msk.nii.gz
│   └── isles_eval_util.py
└── main.py
```
