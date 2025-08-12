## ISIC-Net (ResNet50)
Fine-tuned ResNet50 for binary skin lesion classification, trained on HAM10000 and BCN20000.

### What is this?
ISIC-Net is a transfer-learning model based on ResNet50, fine-tuned to classify dermoscopic images into two classes. It leverages a combined dataset from HAM10000 and BCN20000 for generalization.

### Datasets
- **HAM10000**: Dermatoscopic images of common pigmented skin lesions
- **BCN20000**: Large-scale dermatoscopic dataset for skin lesion analysis

### Model
- **Backbone**: ResNet50 pre-trained on ImageNet
- **Head**: Replaced final FC layer for binary classification
- **Training**: Fine-tuned on merged HAM10000 + BCN20000 splits

### Results
- **Test accuracy**: 88.1%
- **Classification report**:

```text
              precision    recall  f1-score   support

           0       0.86      0.86      0.86      1607
           1       0.90      0.90      0.90      2251

    accuracy                           0.88      3858
   macro avg       0.88      0.88      0.88      3858
weighted avg       0.88      0.88      0.88      3858
```

### Repository contents
- `isic-net.ipynb`: End-to-end notebook for training and evaluation
- `isic-net.pth`: Trained ResNet50 weights

### Quick start
1. Open `isic-net.ipynb`
2. Ensure `isic-net.pth` is available if you want to load the trained weights
3. Run the notebook cells to evaluate or fine-tune further

