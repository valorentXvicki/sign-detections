# HOG + SVM Feature Extraction and Classification

Complete implementation of Histogram of Oriented Gradients (HOG) feature extraction combined with Support Vector Machine (SVM) classification for object detection.

## Features

### HOG Feature Extraction
- Standard HOG descriptor extraction
- Multi-scale HOG extraction
- Batch processing
- HOG visualization
- Histogram computation
- Configurable parameters (window size, cell size, bins)

### SVM Classification
- Multiple kernel support (linear, RBF, polynomial, sigmoid)
- Feature normalization with StandardScaler
- Probability estimation
- Cross-validation support
- Model persistence (save/load)
- Comprehensive evaluation metrics

### Complete Pipeline
- Train HOG + SVM detector
- Batch predictions
- Real-time detection
- Video processing
- Webcam support

## Installation

```bash
pip install scikit-learn opencv-python numpy joblib
```

## Quick Start

### 1. Extract HOG Features

```python
from hog_features import HOGFeatureExtractor
import cv2

# Initialize
hog = HOGFeatureExtractor(win_size=(64, 128))

# Load image
image = cv2.imread('image.jpg')

# Extract features
features = hog.extract(image)
print(f"Feature vector size: {len(features)}")
```

### 2. Train SVM Classifier

```python
from svm_classifier import SVMClassifier

# Initialize classifier
svm = SVMClassifier(kernel='rbf', C=1.0)

# Train
svm.fit(X_train, y_train)

# Evaluate
metrics = svm.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### 3. Train HOG + SVM Detector

```python
from hog_features import HOGFeatureExtractor
from svm_classifier import train_hog_svm_detector

hog = HOGFeatureExtractor(win_size=(64, 128))

# Train detector
classifier, metrics = train_hog_svm_detector(
    positive_images,
    negative_images,
    hog,
    kernel='rbf'
)

print(f"Training accuracy: {metrics['accuracy']:.4f}")
```

### 4. Detect Objects

```python
from detect_hog_svm import HOGSVMPredictor

# Load model
predictor = HOGSVMPredictor('model_directory')

# Detect in image
label, confidence = predictor.detect_image('test.jpg')
print(f"Detection: label={label}, confidence={confidence:.2f}")

# Detect in video
results = predictor.detect_video('video.mp4', 'output.mp4')

# Real-time webcam
predictor.detect_webcam()
```

## Command-Line Usage

### Training

```bash
# Train HOG + SVM detector
python train_hog_svm.py positive_samples/ negative_samples/ \
    --output hog_svm_model \
    --kernel rbf \
    --C 1.0 \
    --test-size 0.2
```

### Detection

```bash
# Detect in image
python detect_hog_svm.py hog_svm_model --source image.jpg

# Detect in video
python detect_hog_svm.py hog_svm_model --source video.mp4 --output output.mp4

# Real-time webcam
python detect_hog_svm.py hog_svm_model --source 0

# Detect in directory
python detect_hog_svm.py hog_svm_model --source test_images/
```

## Module Documentation

### hog_features.py

**HOGFeatureExtractor**
- `extract(image)` - Extract HOG from single image
- `extract_batch(images)` - Extract HOG from batch
- `extract_from_files(paths)` - Extract from file paths
- `extract_from_directory(dir)` - Extract from directory
- `visualize_hog(image)` - Visualize HOG features
- `get_feature_dimension()` - Get feature vector size

**MultiScaleHOG**
- `extract(image)` - Multi-scale HOG extraction
- `extract_batch(images)` - Batch multi-scale extraction

**Functions**
- `compute_hog_histogram()` - Compute HOG histogram

### svm_classifier.py

**SVMClassifier**
- `fit(X, y)` - Train classifier
- `predict(X)` - Predict labels
- `predict_proba(X)` - Predict probabilities
- `evaluate(X, y)` - Evaluate performance
- `cross_validate(X, y)` - Cross-validation
- `save(path)` - Save model
- `load(path)` - Load model

**HOGSVMDetector**
- `detect(image)` - Detect single image
- `detect_batch(images)` - Detect batch
- `save(dir)` - Save detector
- `load(dir)` - Load detector

### train_hog_svm.py

**HOGSVMTrainer**
- `train()` - Train detector
- `evaluate_on_directory()` - Evaluate on test set
- `save_model()` - Save trained model

### detect_hog_svm.py

**HOGSVMPredictor**
- `detect_image()` - Detect in image
- `detect_batch()` - Detect in batch
- `detect_directory()` - Detect in directory
- `detect_video()` - Detect in video
- `detect_webcam()` - Real-time webcam detection

## Training Workflow

1. **Prepare Dataset**
   ```
   dataset/
   ├── positive/
   │   ├── sample1.jpg
   │   ├── sample2.jpg
   │   └── ...
   └── negative/
       ├── sample1.jpg
       ├── sample2.jpg
       └── ...
   ```

2. **Train Model**
   ```bash
   python train_hog_svm.py dataset/positive/ dataset/negative/ \
       --output my_model --kernel rbf
   ```

3. **Evaluate**
   - Results printed to console
   - Model saved to output directory

4. **Use for Detection**
   ```bash
   python detect_hog_svm.py my_model --source test.jpg
   ```

## Detection Result Format

```
Detection: label=1, confidence=0.8542
```

For batch/directory:
```
image1.jpg: label=1, confidence=0.8542
image2.jpg: label=0, confidence=0.7234
...
```

## Evaluation Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)

## Configuration

### HOG Parameters

- **win_size**: Window size (height, width) - default: (64, 128)
- **block_size**: Block size for HOG - default: (16, 16)
- **block_stride**: Stride between blocks - default: (8, 8)
- **cell_size**: Cell size in pixels - default: (8, 8)
- **nbins**: Number of orientation bins - default: 9

### SVM Parameters

- **kernel**: 'linear', 'rbf', 'poly', 'sigmoid' - default: 'rbf'
- **C**: Regularization parameter - default: 1.0
- **gamma**: Kernel coefficient - default: 'scale'

## Performance Optimization

1. **Feature Extraction**
   - Use batch processing for multiple images
   - Multi-scale extraction for different object sizes
   - Reduce image resolution for faster processing

2. **SVM Training**
   - Use linear kernel for faster training
   - Reduce dataset size for initial testing
   - Use cross-validation to find optimal parameters

3. **Detection**
   - Use smaller window size for faster detection
   - Skip frames in video processing
   - Use GPU acceleration if available

## Examples

See `examples_hog_svm.py` for 8 complete examples:

1. Basic HOG extraction
2. Batch extraction
3. Multi-scale HOG
4. HOG visualization
5. SVM training
6. HOG + SVM detector training
7. Cross-validation
8. HOG histogram computation

## Troubleshooting

- **ImportError**: Install required packages: `pip install scikit-learn opencv-python`
- **Out of memory**: Reduce batch size or use smaller images
- **Low accuracy**: Use larger dataset, tune SVM parameters, or try different kernels
- **Slow detection**: Use smaller window size or linear kernel

## Advantages of HOG + SVM

✅ Robust to changes in lighting and object orientation
✅ Relatively fast compared to deep learning
✅ Works well with small datasets
✅ Interpretable features
✅ No GPU required

## Limitations

❌ Hand-crafted features (less flexible than deep learning)
❌ Requires manual parameter tuning
❌ Less accurate than modern deep learning methods
❌ Sensitive to object scale and rotation

## Advanced Usage

### Custom HOG Parameters

```python
hog = HOGFeatureExtractor(
    win_size=(128, 256),
    block_size=(32, 32),
    cell_size=(16, 16),
    nbins=12
)
```

### Custom SVM Parameters

```python
svm = SVMClassifier(
    kernel='poly',
    C=10.0,
    gamma='auto'
)
```

### Multi-Scale Detection

```python
ms_hog = MultiScaleHOG(scales=[(32, 64), (64, 128), (256, 512)])
features = ms_hog.extract(image)
```

## References

- [HOG: Histogram of Oriented Gradients](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- [SVM: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [OpenCV HOG Descriptor](https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html)
