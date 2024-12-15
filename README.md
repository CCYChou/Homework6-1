# HW6-1: Medical Mask Classification Using Transfer Learning

This project demonstrates the use of transfer learning with the VGG16 model for a binary image classification problem. It classifies images into two classes: "with_mask" and "without_mask". This README outlines the project based on the **CRISP-DM** methodology.

---

## 1. Business Understanding

The goal is to build a machine learning model that classifies whether a person is wearing a medical mask in an image. This has applications in public safety, health monitoring, and compliance checks in crowded places.

---

## 2. Data Understanding

### Dataset
- The dataset comprises images categorized into two classes: `with_mask` and `without_mask`.
- If a real dataset is unavailable, the project includes a function to generate a synthetic dataset using random noise and basic augmentation.

### Synthetic Dataset
- Synthetic images are generated programmatically with distinct visual properties to simulate the two classes.

---

## 3. Data Preparation

### Steps
1. **Data Augmentation**: 
   - The `ImageDataGenerator` class is used to apply transformations such as rotation, zooming, shifting, and flipping to improve model generalization.
   
2. **Directory Structure**:
   - Images are organized into directories:
     ```
     medical_mask_dataset/
     ├── with_mask/
     └── without_mask/
     ```

3. **Validation Split**:
   - 80% of the data is used for training, and 20% for validation.

---

## 4. Modeling

### Model Architecture
- **Base Model**: VGG16 pre-trained on ImageNet.
- **Custom Layers**:
  - Global average pooling
  - Dense layers with ReLU activations
  - Batch normalization
  - Dropout layers to prevent overfitting
  - Output layer with softmax activation for binary classification.

### Transfer Learning
- The base model layers can be fine-tuned for better performance.
- The model is compiled with `Adam` optimizer and `categorical_crossentropy` loss.

---

## 5. Evaluation

### Metrics
- Accuracy and loss are tracked for both training and validation datasets.

### Visualization
- Training history is plotted to show:
  - Training and validation accuracy over epochs.
  - Training and validation loss over epochs.

### Example Test
- A pre-trained model can classify images from a URL input, returning the predicted class and confidence score.

---

## 6. Deployment

### Usage Instructions
1. **Dataset Setup**:
   - If no dataset is available, synthetic data is generated automatically.
   - To use a custom dataset, organize it as follows:
     ```
     medical_mask_dataset/
     ├── with_mask/
     └── without_mask/
     ```

2. **Run the Program**:
   - For Python script (`hw6_1.py`):
     ```
     python hw6_1.py
     ```
   - For Notebook (`HW6_1.ipynb`):
     Open and execute the cells sequentially in a Colab or Jupyter environment.

3. **Test with Image URL**:
   - Input an image URL when prompted to classify it.

### Example
```python
image_url = "https://example.com/mask_image.jpg"
test_image(image_url, model, train_dataset.classes)
