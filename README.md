# Diffusion-Based Data Augmentation for Class Imbalance

Evaluates the effect of synthetic data augmentation using a pretrained 
class-conditional DDPM on minority-class accuracy in CIFAR-10.

## Results
Minority class (cat) accuracy improved from 0.52 (baseline) to 0.60 
(peak at 1.0x synthetic fraction), with diminishing returns beyond that.

## How to run
pip install -r requirements.txt
python preprocess.py
