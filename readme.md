# Face Mask Detection using VGG16 Transfer Learning

## 1. Business Understanding

### Project Goal
Build a face mask detection system using transfer learning with VGG16 pre-trained model for detecting whether a person is wearing a mask or not.

### Objectives
- Achieve high accuracy (>90%) in mask detection
- Provide real-time predictions from image URLs
- Create a reusable and maintainable codebase

### Success Criteria
- Model accuracy > 90%
- Quick inference time (< 2 seconds)
- Reliable predictions across different face angles and lighting conditions

## 2. Data Understanding

### Dataset
- Source: GitHub medical mask dataset
- URL: https://github.com/prajnasb/observations
- Classes:
  - with_mask: People wearing masks
  - without_mask: People without masks
- Format: JPG images
- Input size: 224x224 pixels (after preprocessing)

### Data Collection
```python
def download_dataset():
    base_url = "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/data"
    classes = ['with_mask', 'without_mask']
    ...
```

## 3. Data Preparation

### Installation
```bash
pip install tensorflow pillow requests matplotlib wget numpy
```

### Data Preprocessing
- Image resizing to 224x224
- Normalization (scaling pixel values to [0,1])
- Data augmentation:
  - Rotation (±20 degrees)
  - Width/height shifts (20%)
  - Shear transformation
  - Zoom range
  - Horizontal flips

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
```

## 4. Modeling

### Model Architecture
- Base: VGG16 (pre-trained on ImageNet)
- Additional layers:
  - Flatten
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (1 unit, Sigmoid)

```python
def build_model():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    ...
```

### Training Configuration
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 30
- Batch size: 32

## 5. Evaluation

### Performance Metrics
- Training accuracy
- Validation accuracy
- Loss curves
- Per-prediction confidence scores

### Visualization
```python
def plot_training_history(history):
    # Plot accuracy and loss curves
    ...

def test_image(image_url, model):
    # Display prediction results
    ...
```

## 6. Deployment

### Running the Project
1. Clone repository:
```bash
git clone [repository-url]
cd face-mask-detection
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```

### Testing Model
```python
# Example usage
image_url = "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/data/with_mask/1.jpg"
test_image(image_url, model)
```

### Directory Structure
```
project/
├── data/
│   ├── with_mask/
│   └── without_mask/
├── models/
│   └── mask_detection_model.h5
├── main.py
└── README.md
```

## Usage Examples

### Training New Model
```python
# Run main script
if __name__ == "__main__":
    main()
```

### Making Predictions
```python
# Load saved model
model = tf.keras.models.load_model('mask_detection_model.h5')

# Test with image
image_url = input("Enter image URL: ")
test_image(image_url, model)
```

## Model Limitations
- Works best with front-facing faces
- Requires good lighting conditions
- May be affected by unusual mask types
- Limited to binary classification (mask/no mask)

## Future Improvements
- [ ] Add support for multiple face detection
- [ ] Implement real-time video processing
- [ ] Add more mask types classification
- [ ] Improve model robustness to lighting conditions
- [ ] Add model interpretation visualizations

## Requirements
- Python 3.7+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)
- Minimum 4GB RAM

## License
This project is licensed under the MIT License.
