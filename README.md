## HamNet (ResNet50 & DenseNet121)
Fine-tuned ResNet50 and DenseNet121 for skin lesion classification on HAM10000.

### What is this?
HamNet provides two transfer-learning models (ResNet50 and DenseNet121), fine-tuned to classify dermoscopic images into two classes on the HAM10000 dataset.

### Datasets
- **HAM10000**: Dermatoscopic images of common pigmented skin lesions

### Model
- **Backbones**: ResNet50 and DenseNet121 pre-trained on ImageNet
- **Head**: FiLM-based fusion head. Metadata (sex, age, site) is passed through a gated FiLM to produce per-feature scales (γ) that modulate the backbone feature vector via element-wise multiplication.
- **Fusion**: A small MLP also embeds the metadata; this embedding is concatenated with the FiLM-gated backbone features and fed into the final classifier.
- **Training**: Fine-tuned on HAM10000 stratified splits

### Results

| Model | Test accuracy (HAM10000) |
| --- | --- |
| ResNet50 | 91.3% |
| DenseNet121 | 90.2% |

### Fine-tuned models
- ResNet50: [Ham-ResNet](https://huggingface.co/galactixx/Ham-ResNet)
- DenseNet121: [Ham-DenseNet](https://huggingface.co/galactixx/Ham-DenseNet)

### Repository contents
- `hamnet/train_resnet.py`: Train ResNet50 with progressive unfreezing
- `hamnet/test_resnet.py`: Evaluate a pretrained ResNet50 on the test split
- `hamnet/train_densenet.py`: Train DenseNet121 with progressive unfreezing
- `hamnet/test_densenet.py`: Evaluate a pretrained DenseNet121 on the test split

### Quick start
1. Download HAM10000 from the ISIC Archive: go to [HAM10000 collection](https://api.isic-archive.com/collections/212/), click "Actions" → "Download Collection".
2. Unzip the download into `data/HAM10000/` so that images and metadata reside under this folder.
3. Evaluate pretrained models:
   - ResNet50:
     ```bash
     python hamnet/test_resnet.py
     ```
   - DenseNet121:
     ```bash
     python hamnet/test_densenet.py
     ```
4. Train from ImageNet-pretrained backbones (optional):
   - ResNet50:
     ```bash
     python hamnet/train_resnet.py
     ```
   - DenseNet121:
     ```bash
     python hamnet/train_densenet.py
     ```

