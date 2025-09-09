# Integration Guide: MLP Classifier for Contrast Phase Detection

## Overview

This guide helps you integrate the new MLP-based approach with your existing contrast phase detection codebase, replacing the LDA approach with an end-to-end trainable neural network.

## Key Advantages Over LDA

- ✅ **End-to-end trainable**: No separate feature extraction step
- ✅ **Better gradient flow**: Enables proper saliency map generation
- ✅ **Attention mechanisms**: Improves interpretability
- ✅ **Flexible architecture**: Customizable MLP heads
- ✅ **3D saliency maps**: Direct visualization on 3D volumes

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```

### 2. Basic Usage

```python
from contrast_phase_mlp_classifier import create_model_and_trainer

# Create model
model, trainer = create_model_and_trainer(
    encoder_type='medvit',
    encoder_config={
        'model_size': 'small',
        'latent_dim': 256,
        'max_slices': 32
    },
    n_classes=5
)

# Train (assuming you have data loaders)
trainer.train(train_loader, val_loader, num_epochs=50)

# Generate saliency maps
saliency_results = model.generate_3d_saliency_visualization(test_volume)
```

### 3. Full Experiment

```python
from train_contrast_phase_mlp import ContrastPhaseMLPExperiment, create_default_config

# Configure experiment
config = create_default_config()
config.update({
    'data_path': 'your/data/path',
    'batch_size': 4,
    'num_epochs': 50
})

# Run experiment
experiment = ContrastPhaseMLPExperiment(config)
results = experiment.run_experiment()
```

## Detailed Integration Steps

### Step 1: Replace Your LDA Pipeline

**Old LDA approach:**
```python
# Old way - separate feature extraction and LDA
features = extract_features_from_encoder(encoder, data_loader)
lda = LinearDiscriminantAnalysis()
lda.fit(features, labels)
predictions = lda.predict(test_features)
```

**New MLP approach:**
```python
# New way - end-to-end training
model = ContrastPhaseMLPClassifier(encoder, encoder_name, n_classes=5)
trainer = ContrastPhaseTrainer(model)
trainer.train(train_loader, val_loader)
predictions, probabilities = model.predict(test_data)
```

### Step 2: Update Data Pipeline

Your existing data pipeline should work with minimal changes:

```python
# Your existing data preparation (no changes needed)
train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
    data_path, labels_csv, validation_split=0.2
)

train_loader = prepare_data(train_data_dicts, batch_size=4)
val_loader = prepare_data(val_data_dicts, batch_size=4)

# Use with new MLP trainer
trainer.train(train_loader, val_loader)
```

### Step 3: Generate Saliency Maps

```python
# Load your 3D volume
volume = torch.randn(1, 1, 128, 128, 128)  # Your actual volume

# Generate GradCAM saliency map
saliency_map, pred_class, confidence = model.generate_gradcam(volume)

# Generate comprehensive visualization
saliency_results = model.generate_3d_saliency_visualization(
    volume, 
    save_path='saliency_visualization.png'
)
```

## Configuration Options

### Encoder Configurations

**MedViT:**
```python
medvit_config = {
    'model_size': 'small',  # 'small', 'base', 'large'
    'pretrained_path': 'path/to/weights.pth',
    'latent_dim': 256,
    'aggregation_method': 'lstm',  # 'lstm', 'attention', 'mean'
    'max_slices': 32
}
```

**TimmViT:**
```python
timm_config = {
    'model_name': 'vit_small_patch16_224',
    'pretrained': True,
    'latent_dim': 256,
    'max_slices': 32
}
```

**DinoV3:**
```python
dinov3_config = {
    'model_size': 'small',  # 'small', 'base', 'large'
    'pretrained': True,
    'latent_dim': 256,
    'max_slices': 32
}
```

### MLP Configurations

```python
mlp_config = {
    'hidden_dims': [512, 256, 128],  # Hidden layer sizes
    'dropout_rate': 0.3,             # Dropout for regularization
    'use_attention': True,           # Enable attention mechanism
    'attention_heads': 8             # Number of attention heads
}
```

### Training Configurations

```python
training_config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'freeze_encoder': False  # Set True for feature extraction only
}
```

## Migration from Your Current Code

### 1. Update phase_detector.py

Replace the LDA-based `ContrastPhaseClassifier` usage:

```python
# Old usage in phase_detector.py
from phase_detector import ContrastPhaseClassifier
classifier = ContrastPhaseClassifier(encoder, encoder_name)
classifier.fit(train_features, train_labels)

# New usage
from contrast_phase_mlp_classifier import ContrastPhaseMLPClassifier, ContrastPhaseTrainer
model = ContrastPhaseMLPClassifier(encoder, encoder_name, n_classes=5)
trainer = ContrastPhaseTrainer(model)
trainer.train(train_loader, val_loader)
```

### 2. Update Evaluation Code

```python
# Old evaluation
test_results = classifier.evaluate(test_features, test_labels)

# New evaluation
val_loss, val_acc, detailed_metrics = trainer.validate_epoch(test_loader)
```

### 3. Update Visualization Code

```python
# Old visualization (limited)
classifier.plot_lda_projection(features, labels)

# New visualization (comprehensive)
saliency_results = model.generate_3d_saliency_visualization(volume)
trainer.plot_training_history()
```

## Advanced Features

### 1. Custom Encoder Integration

```python
class CustomEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        # Your custom architecture
        
    def forward(self, x):
        # Your forward pass
        return features

# Use with MLP classifier
model = ContrastPhaseMLPClassifier(
    encoder=CustomEncoder(),
    encoder_name="CustomEncoder",
    n_classes=5
)
```

### 2. Multi-Scale Saliency Maps

```python
def generate_multiscale_saliency(model, volume, scales=[0.5, 1.0, 1.5]):
    """Generate saliency maps at multiple scales"""
    saliency_maps = []
    
    for scale in scales:
        # Resize volume
        scaled_size = [int(s * scale) for s in volume.shape[2:]]
        scaled_volume = F.interpolate(volume, size=scaled_size, mode='trilinear')
        
        # Generate saliency map
        saliency_map, _, _ = model.generate_gradcam(scaled_volume)
        
        # Resize back to original size
        original_size = volume.shape[2:]
        saliency_map_resized = F.interpolate(
            torch.tensor(saliency_map).unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode='trilinear'
        ).squeeze().numpy()
        
        saliency_maps.append(saliency_map_resized)
    
    # Combine maps (average)
    combined_map = np.mean(saliency_maps, axis=0)
    return combined_map, saliency_maps
```

### 3. Ensemble Predictions

```python
def ensemble_prediction(models, volume):
    """Get ensemble prediction from multiple models"""
    predictions = []
    probabilities = []
    
    for model in models:
        pred, prob = model.predict(volume)
        predictions.append(pred)
        probabilities.append(prob)
    
    # Average probabilities
    avg_prob = torch.mean(torch.stack(probabilities), dim=0)
    ensemble_pred = torch.argmax(avg_prob, dim=1)
    
    return ensemble_pred, avg_prob
```

## Performance Optimization

### 1. Memory Optimization

```python
# Use gradient checkpointing for large models
model = ContrastPhaseMLPClassifier(
    encoder=encoder,
    encoder_name="OptimizedEncoder",
    n_classes=5
)

# Enable gradient checkpointing if available
if hasattr(model.encoder, 'gradient_checkpointing_enable'):
    model.encoder.gradient_checkpointing_enable()
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

# In ContrastPhaseTrainer, use automatic mixed precision
scaler = GradScaler()

# Modified training loop
with autocast():
    logits = model(images)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Efficient Saliency Generation

```python
# Batch saliency map generation
def generate_batch_saliency(model, volumes, batch_size=4):
    """Generate saliency maps for multiple volumes efficiently"""
    all_saliency_maps = []
    
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size]
        batch_saliency = []
        
        for volume in batch:
            saliency_map, _, _ = model.generate_gradcam(volume.unsqueeze(0))
            batch_saliency.append(saliency_map)
        
        all_saliency_maps.extend(batch_saliency)
    
    return all_saliency_maps
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   config['batch_size'] = 2
   
   # Use gradient accumulation
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Poor Saliency Map Quality**
   ```python
   # Don't freeze encoder
   freeze_encoder = False
   
   # Use attention aggregation
   aggregation_method = 'attention'
   
   # Enable MLP attention
   mlp_use_attention = True
   ```

3. **Slow Training**
   ```python
   # Reduce spatial size for initial experiments
   spatial_size = [64, 64, 64]
   
   # Use fewer slices
   max_slices = 16
   
   # Freeze encoder initially
   freeze_encoder = True
   ```

4. **Model Not Learning**
   ```python
   # Check learning rate
   learning_rate = 1e-3  # Start higher
   
   # Reduce regularization
   weight_decay = 1e-5
   dropout_rate = 0.1
   
   # Ensure labels are correct format
   labels = labels.long()  # For CrossEntropyLoss
   ```

## Testing and Validation

### Unit Tests

```python
def test_model_creation():
    """Test model creation with different encoders"""
    for encoder_type in ['medvit', 'timm_vit', 'dinov3']:
        model, trainer = create_model_and_trainer(
            encoder_type=encoder_type,
            encoder_config={'latent_dim': 128},
            n_classes=5
        )
        assert model is not None
        assert trainer is not None

def test_saliency_generation():
    """Test saliency map generation"""
    # Create dummy model and data
    model = create_dummy_model()
    volume = torch.randn(1, 1, 32, 32, 32)
    
    # Test GradCAM
    saliency_map, pred, prob = model.generate_gradcam(volume)
    assert saliency_map.shape == volume.shape[2:]
    assert 0 <= pred < 5
    assert 0 <= prob <= 1
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete training pipeline"""
    # Create small dataset
    dummy_data = create_dummy_dataset()
    
    # Create model
    model, trainer = create_model_and_trainer(
        encoder_type='timm_vit',
        encoder_config={'latent_dim': 64},
        n_classes=5
    )
    
    # Train for few epochs
    history = trainer.train(dummy_data, dummy_data, num_epochs=2)
    
    # Check training completed
    assert len(history['train_loss']) == 2
    assert len(history['train_acc']) == 2
```

## Migration Checklist

- [ ] Install new dependencies
- [ ] Update import statements
- [ ] Replace LDA classifier with MLP classifier
- [ ] Update training loop
- [ ] Test saliency map generation
- [ ] Validate results on small dataset
- [ ] Run full experiment
- [ ] Compare results with LDA baseline
- [ ] Document performance improvements

## Next Steps

1. **Start with Quick Test**: Use the quick training example to validate setup
2. **Compare with LDA**: Run both approaches on same data to compare
3. **Optimize Configuration**: Tune hyperparameters for your specific data
4. **Scale Up**: Move to full dataset and longer training
5. **Analyze Saliency Maps**: Validate medical relevance of highlighted regions

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Run the usage examples to validate setup
3. Use smaller datasets/models for debugging
4. Check tensor shapes and data types
5. Monitor GPU memory usage

The new MLP approach should provide better performance and more interpretable saliency maps compared to the LDA baseline while maintaining compatibility with your existing data pipeline.