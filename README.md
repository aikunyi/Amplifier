# Amplifier

The repo is the official implementation for the paper: "Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting".



## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n Amplifier python=3.8
conda activate Amplifier
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for Amplifier can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. 
Create a separate folder named ```./dataset``` and place all the CSV files in this directory. 
**Note**: Place the CSV files directly into this directory, such as "./dataset/ETT-small/ETTh1.csv"

### Training Example

You can reproduce the experiment results as the following examples:

```
bash ./scripts/ETTm1.sh
bash ./scripts/ETTh1.sh
bash ./scripts/ECL.sh
```

## Acknowledgement

We appreciate the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/lss-1138/SparseTSF

https://github.com/plumprc/RTSF

https://github.com/cure-lab/LTSF-Linear

https://github.com/aikunyi/FreTS

https://github.com/VEWOXIC/FITS

https://github.com/chenzRG/Fredformer

https://github.com/thuml/iTransformer

https://github.com/yuqinie98/PatchTST

https://github.com/thuml/Nonstationary_Transformers

https://github.com/thuml/TimesNet

https://github.com/thuml/Autoformer

