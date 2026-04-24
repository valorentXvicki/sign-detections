# Road Sign YAML Dataset Configuration

Complete guide to creating and using road sign YAML dataset files for YOLO object detection.

## Files Created

1. **roadsigns.yaml** - Standard European road signs (43 classes)
2. **roadsigns_utils.py** - Dataset management utilities
3. **dataset_manager.py** - Advanced dataset manager with validation
4. **roadsigns_templates.py** - Predefined templates (European, German, US)
5. **examples_roadsigns.py** - Usage examples
6. **create_advanced_yaml.py** - Advanced YAML generators

## YAML File Structure

### Basic Format

```yaml
path: /path/to/road_signs_dataset
train: images/train
val: images/val
test: images/test

nc: 43
names: ['Speed limit 20', 'Speed limit 30', ..., 'Bicycle lane']
```

### Required Fields

- **path**: Root path to dataset directory
- **train**: Path to training images (relative to root)
- **val**: Path to validation images (relative to root)
- **test**: Path to test images (relative to root)
- **nc**: Number of classes
- **names**: List of class names (order matches class IDs)

## Dataset Structure

```
road_signs_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt  (YOLO format)
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

## Road Sign Classes

### Speed Limit Signs (0-7)
- Speed limit 20, 30, 50, 60, 70, 80, 100, 120 km/h

### Prohibition Signs (8-18)
- No passing, No passing trucks, No entry, No vehicles
- No horns, No parking, No stopping

### Warning Signs (19-29)
- Pedestrian crossing, Children crossing, Bicycle crossing
- Slippery road, Road work, Traffic signals
- Bumpy road, Uneven road, Road narrows
- Falling rocks, Animal crossing

### Mandatory Signs (10-11, 31-35, 39-42)
- Priority road, Right of way, Yield, Stop
- Turn left/right, Keep left/right, Roundabout
- One way, Pedestrian zone, Bicycle lane

### Information Signs (30, 36-38)
- End of speed limit, End no passing
- End speed 80, End all restrictions

## Creating YAML Files

### 1. From Template

```python
from roadsigns_templates import create_yaml_from_template, get_template

template = get_template('european')  # or 'german', 'us'
create_yaml_from_template(
    template,
    'roadsigns.yaml',
    dataset_path='/dataset/roadsigns'
)
```

### 2. Custom Signs

```python
from roadsigns_templates import create_custom_yaml

signs = ['Stop', 'Yield', 'Speed limit', 'No entry']
create_custom_yaml(signs, 'my_roadsigns.yaml')
```

### 3. Advanced with Metadata

```python
from create_advanced_yaml import create_advanced_roadsigns_yaml

create_advanced_roadsigns_yaml('roadsigns_advanced.yaml')
```

### 4. Multilingual

```python
from create_advanced_yaml import create_multilingual_roadsigns_yaml

create_multilingual_roadsigns_yaml('roadsigns_multilingual.yaml')
```

## Using Dataset Manager

### Load and Validate

```python
from dataset_manager import RoadSignDataset

dataset = RoadSignDataset('roadsigns.yaml')

# Get information
print(f"Classes: {dataset.get_num_classes()}")
print(f"Names: {dataset.get_class_names()}")

# Validate
validation = dataset.validate_dataset()
print(f"Valid: {validation['valid']}")

# Print info
dataset.print_info()

# Export stats
dataset.export_stats('stats.json')
```

### Access Information

```python
# Get class by ID
class_name = dataset.get_class_name(0)  # 'Speed limit 20'

# Get ID by name
class_id = dataset.get_class_id('Stop')  # 13

# Get statistics
stats = dataset.get_statistics()
```

## Training with YAML

### YOLOv8 Training

```bash
yolo detect train data=roadsigns.yaml model=yolov8s.pt epochs=100 imgsz=640
```

### Python Training

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='roadsigns.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

## Validation

```python
from dataset_manager import RoadSignDataset

dataset = RoadSignDataset('roadsigns.yaml')
validation = dataset.validate_dataset()

print(f"Validation Results:")
print(f"  Valid: {validation['valid']}")
print(f"  Errors: {validation['errors']}")
print(f"  Warnings: {validation['warnings']}")
print(f"  Statistics: {validation['statistics']}")
```

## Available Templates

### European (43 classes)
Standard European road signs used across EU countries

### German (63 classes)
Extended German traffic sign set with more detailed classifications

### US (22 classes)
Common US road signs (simplified set)

### Custom
Define your own sign classes

## Command-Line Tools

### Create YAML

```bash
# Using templates
python roadsigns_templates.py

# Advanced YAML
python create_advanced_yaml.py

# Examples
python examples_roadsigns.py
```

### Validate YAML

```python
from dataset_manager import RoadSignDataset

dataset = RoadSignDataset('roadsigns.yaml')
validation = dataset.validate_dataset()
```

## Examples

See `examples_roadsigns.py` for complete examples:

1. Create from template
2. Load and validate dataset
3. Create custom YAML
4. Access dataset information
5. Export statistics
6. Compare templates

## Tips & Best Practices

1. **Directory Structure**
   - Keep images and labels organized
   - Use consistent naming conventions
   - Ensure labels directory matches images directory

2. **Class Names**
   - Use consistent, descriptive names
   - Match order with label indices
   - Keep names short but clear

3. **Path Configuration**
   - Use absolute paths or relative to root
   - Ensure paths are accessible
   - Use forward slashes in YAML

4. **Validation**
   - Always validate YAML before training
   - Check image counts in each split
   - Verify class count matches names

5. **Dataset Balance**
   - Ensure good distribution across classes
   - Handle class imbalance if needed
   - Monitor class-wise metrics

## Troubleshooting

- **Path not found**: Check absolute/relative paths
- **Class mismatch**: Verify `nc` matches `len(names)`
- **Missing splits**: Ensure train/val/test directories exist
- **Invalid YAML**: Check YAML syntax with `yaml.safe_load()`

## Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com)
- [Road Sign Datasets](https://www.kaggle.com/datasets)
- [Traffic Sign Recognition](https://github.com/topics/traffic-sign-recognition)
