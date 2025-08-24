# CT Contrast Phase Generation

A deep learning framework for generating different contrast phases in CT scans using advanced encoder-generator architectures with disentangled representations.

## 🚀 Overview

This project implements a sophisticated pipeline that can transform CT scans between different contrast phases (non-contrast, arterial, venous, delayed) using state-of-the-art medical vision transformers and generative adversarial networks.

### Key Features

- **Multiple Encoder Architectures**: MedViT, Timm ViT, ResNet3D, Simple 3D CNN, Hybrid CNN-Transformer
- **Disentangled Representation Learning**: Separates anatomical content from contrast phase information
- **Advanced Training Pipeline**: Sequential training with encoder pretraining, phase detection, and adversarial generation
- **Comprehensive Evaluation**: Image quality metrics including PSNR, SSIM, MS-SSIM, NMSE, NCC, and Mutual Information
- **Feature Visualization**: t-SNE and PCA visualizations for understanding learned representations
- **Early Stopping & Monitoring**: Advanced early stopping with gradient norm tracking and model-specific stopping criteria

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Feature Visualization](#feature-visualization)
  - [Benchmarking](#benchmarking)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)

## 🛠 Installation

### Requirements

```bash
# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
nibabel>=3.2.0
pandas>=1.3.0
tqdm>=4.62.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Medical imaging
monai>=1.0.0
SimpleITK>=2.2.0

# Machine learning
scikit-learn>=1.1.0
scipy>=1.7.0

# Optional: For advanced models
timm>=0.6.0  # For Timm ViT encoder
```

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/ct-contrast-generation.git
cd ct-contrast-generation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install MedViT** (if using MedViT encoder):
```bash
git clone https://github.com/Omid-Nejati/MedViT.git
# Place MedViT folder in your project directory
```

4. **Download pretrained weights** (optional):
```bash
# Download pretrained MedViT weights
wget https://example.com/pretrained_medvit_small.pth
```

## 📁 Dataset Preparation

### Directory Structure

Organize your CT data as follows:

```
data/
├── labels.csv
├── scan_001/
│   ├── non_contrast.nii.gz
│   ├── arterial.nii.gz
│   ├── venous.nii.gz
│   └── delayed.nii.gz
├── scan_002/
│   ├── non_contrast.nii.gz
│   ├── arterial.nii.gz
│   └── venous.nii.gz
└── ...
```

### Labels CSV Format

Create a `labels.csv` file mapping series to contrast phases:

```csv
SeriesInstanceUID,Label
series_001_nc,NC
series_001_art,A
series_001_pv,PV
series_001_del,D
```

**Phase Labels**:
- `NC` / `Non-contrast` / `Pre` → 0
- `A` / `Arterial` / `Art` → 1  
- `PV` / `Venous` / `Portal Venous` → 2
- `D` / `Delayed` / `Delay` → 3

### Data Validation

Validate your dataset before training:

```bash
python -c "
from data import test_data_loading
test_data_loading('data/', 'data/labels.csv')
"
```

## 🎯 Usage

### Training

#### Basic Training

```bash
python main.py --mode train \
               --data_path data/ \
               --encoder medvit \
               --batch_size 2 \
               --epochs 150 \
               --spatial_size 128 128 128 \
               --latent_dim 256
```

#### Advanced Training with MedViT

```bash
python main.py --mode train \
               --data_path data/ \
               --encoder medvit \
               --medvit_size small \
               --medvit_pretrained_path pretrained_medvit_small.pth \
               --aggregation_method lstm \
               --max_slices 32 \
               --batch_size 2 \
               --epochs 150 \
               --mixed_precision \
               --apply_registration
```

#### Training with Different Encoders

```bash
# Simple 3D CNN
python main.py --mode train --encoder simple_cnn --latent_dim 256

# Timm ViT with pretrained weights
python main.py --mode train --encoder timm_vit --timm_pretrained \
               --timm_model_name vit_small_patch16_224

# ResNet3D
python main.py --mode train --encoder resnet3d --latent_dim 512

# Hybrid CNN-Transformer
python main.py --mode train --encoder hybrid --latent_dim 256
```

### Inference

Generate contrast phases from trained models:

```bash
python main.py --mode inference \
               --checkpoint checkpoints/final_checkpoint.pth \
               --input_volume patient_scan_arterial.nii.gz \
               --input_phase 1 \
               --target_phase 2 \
               --output_path generated_venous_phase.nii.gz \
               --spatial_size 128 128 128
```

**Phase IDs**:
- 0: Non-contrast
- 1: Arterial  
- 2: Venous
- 3: Delayed

### Feature Visualization

Analyze learned representations:

```bash
python feature_visualization.py \
               --data_path data/ \
               --encoder medvit \
               --medvit_size small \
               --batch_size 4 \
               --output_dir visualizations/
```

This generates:
- t-SNE and PCA plots for each encoder
- Comparison plots between different encoders
- Clustering quality metrics (silhouette scores)

### Benchmarking

Measure inference performance:

```bash
python main.py --mode benchmark \
               --checkpoint checkpoints/final_checkpoint.pth \
               --spatial_size 128 128 128
```

## 🏗 Model Architectures

### Encoders

1. **MedViT Encoder** (`medViT_encoder.py`)
   - Medical Vision Transformer for 3D volumes
   - Slice-by-slice processing with feature aggregation
   - Supports LSTM, attention, mean, or max aggregation
   - Pretrained weights available

2. **Timm ViT Encoder** (`models.py`)
   - Uses timm library pretrained ViTs
   - 2D slice processing with 3D aggregation
   - Adaptive slice sampling strategies

3. **Simple 3D CNN** (`models.py`)
   - Lightweight 3D convolutional encoder
   - No external dependencies
   - Good baseline performance

4. **ResNet3D** (`models.py`)
   - 3D adaptation of ResNet architecture
   - Skip connections for better gradient flow

5. **Hybrid CNN-Transformer** (`models.py`)
   - Combines CNN feature extraction with self-attention
   - Balanced approach between efficiency and performance

### Generator & Discriminator

- **Generator**: 3D deconvolutional network with phase embedding
- **Discriminator**: PatchGAN-style 3D CNN for realistic generation
- **Phase Detector**: Classifies contrast phases for disentanglement

## 🔄 Training Pipeline

The training follows a sophisticated 3-phase approach:

### Phase 1: Encoder-Generator Pretraining (30 epochs)
- Trains encoder and generator with reconstruction loss
- Establishes basic volume encoding capabilities
- Uses MSE + L1 loss with regularization

### Phase 2: Phase Detector Training (40 epochs)
- Freezes encoder, trains phase detector
- Class-weighted loss for imbalanced datasets
- Advanced early stopping with oscillation detection

### Phase 3: Disentangled Generation (80 epochs)
- Adversarial training with gradient reversal
- Combines reconstruction, adversarial, and disentanglement losses
- Multi-scale perceptual loss for better image quality

### Key Training Features

- **Mixed Precision Training**: Faster training with lower memory usage
- **Gradient Clipping**: Prevents gradient explosion
- **Advanced Early Stopping**: Model-specific stopping criteria
- **Learning Rate Scheduling**: Adaptive learning rates per phase
- **Comprehensive Validation**: Image quality metrics tracking

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--encoder` | Encoder architecture | `medvit` |
| `--spatial_size` | Input volume dimensions | `128 128 128` |
| `--latent_dim` | Latent representation size | `256` |
| `--batch_size` | Training batch size | `2` |
| `--epochs` | Total training epochs | `150` |
| `--mixed_precision` | Enable mixed precision | `False` |

### MedViT Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--medvit_size` | Model size (tiny/small/base) | `small` |
| `--aggregation_method` | Feature aggregation | `lstm` |
| `--max_slices` | Maximum slices to process | `32` |
| `--slice_sampling` | Sampling strategy | `uniform` |

### Timm ViT Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--timm_model_name` | Timm model name | `vit_small_patch16_224` |
| `--timm_pretrained` | Use pretrained weights | `False` |

## 📊 Results

### Evaluation Metrics

The framework provides comprehensive evaluation:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MS-SSIM**: Multi-Scale SSIM
- **NMSE**: Normalized Mean Squared Error
- **NCC**: Normalized Cross Correlation
- **Mutual Information**: Information theory metric

### Model Performance

Results are saved in:
- `checkpoints/enhanced_training_metrics.csv`
- `checkpoints/validation_metrics.csv`
- `checkpoints/training_summary.json`

### Visualization Outputs

Feature visualization generates:
- Individual encoder t-SNE/PCA plots
- Comparative analysis between encoders
- Clustering quality metrics
- Phase separation analysis

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or spatial size
   --batch_size 1 --spatial_size 96 96 96
   ```

2. **MedViT Import Error**
   ```bash
   # Ensure MedViT folder is in project directory
   git clone https://github.com/Omid-Nejati/MedViT.git
   ```

3. **Data Loading Errors**
   ```bash
   # Validate dataset structure
   python -c "from data import validate_dataset; validate_dataset('data/', 'data/labels.csv')"
   ```

4. **Memory Issues during Feature Visualization**
   ```bash
   # Use smaller batch size
   python feature_visualization.py --batch_size 1
   ```

### Performance Tips

1. **Use Mixed Precision**: Add `--mixed_precision` for 30-50% speedup
2. **Optimize Batch Size**: Find maximum batch size that fits in GPU memory
3. **Enable Registration**: Use `--apply_registration` for better alignment
4. **Use Cached Data**: Add `--skip_prep` to reuse prepared datasets

## 📝 File Structure

```
├── main.py                    # Main entry point
├── training.py               # Training pipeline
├── models.py                 # Model architectures  
├── medViT_encoder.py         # MedViT implementation
├── data.py                   # Data loading and preparation
├── utils.py                  # Utility functions
├── feature_visualization.py  # Feature analysis tools
├── inference.py             # Inference functions (implied)
├── early_stopping_system.py # Early stopping utilities (implied)
├── image_quality_metrics.py # Evaluation metrics (implied)
└── README.md                # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation



## 🙏 Acknowledgments

- MedViT team for the medical vision transformer architecture
- MONAI team for medical imaging utilities
- PyTorch team for the deep learning framework
- timm library for pretrained vision models

---

**Note**: This project is designed for research purposes. Ensure proper validation and regulatory approval before clinical use.