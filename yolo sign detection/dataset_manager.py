"""
Road sign dataset management and validation
"""

import yaml
from pathlib import Path
from typing import Dict, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadSignDataset:
    """Manager for road sign dataset"""
    
    def __init__(self, yaml_path: str):
        """
        Initialize dataset manager
        
        Args:
            yaml_path: Path to dataset YAML file
        """
        self.yaml_path = Path(yaml_path)
        self.config = self.load_config()
        self.dataset_root = self.yaml_path.parent / self.config.get('path', '.')
        
        logger.info(f"Initialized dataset from {yaml_path}")
    
    def load_config(self) -> Dict:
        """Load dataset configuration"""
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config with {config['nc']} classes")
        return config
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.config['names']
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return self.config['nc']
    
    def get_class_id(self, class_name: str) -> int:
        """Get class ID by name"""
        try:
            return self.config['names'].index(class_name)
        except ValueError:
            logger.warning(f"Class not found: {class_name}")
            return None
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID"""
        if 0 <= class_id < len(self.config['names']):
            return self.config['names'][class_id]
        
        logger.warning(f"Invalid class ID: {class_id}")
        return None
    
    def validate_dataset(self) -> Dict:
        """
        Validate dataset structure
        
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required paths
        required_keys = ['train', 'val', 'test', 'nc', 'names']
        for key in required_keys:
            if key not in self.config:
                results['errors'].append(f"Missing key: {key}")
                results['valid'] = False
        
        # Check number of classes matches names
        if self.config['nc'] != len(self.config['names']):
            results['errors'].append(
                f"Class count mismatch: nc={self.config['nc']}, "
                f"names={len(self.config['names'])}"
            )
            results['valid'] = False
        
        # Count images in each split
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_root / self.config[split]
            if split_path.exists():
                image_count = len(list(split_path.glob('*.[jJ][pP][gG]')))
                image_count += len(list(split_path.glob('*.[pP][nN][gG]')))
                results['statistics'][split] = image_count
            else:
                results['warnings'].append(f"Directory not found: {split_path}")
        
        if results['valid']:
            logger.info("Dataset validation passed")
        else:
            logger.error("Dataset validation failed")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names(),
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_root / self.config[split]
            if split_path.exists():
                images = list(split_path.glob('*.[jJ][pP][gG]'))
                images += list(split_path.glob('*.[pP][nN][gG]'))
                stats['splits'][split] = len(images)
            else:
                stats['splits'][split] = 0
        
        stats['total_images'] = sum(stats['splits'].values())
        
        return stats
    
    def export_stats(self, output_path: str):
        """Export statistics to JSON"""
        stats = self.get_statistics()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Exported statistics to {output_path}")
    
    def print_info(self):
        """Print dataset information"""
        print("\n" + "="*70)
        print("ROAD SIGN DATASET INFORMATION")
        print("="*70)
        
        print(f"\nDataset path: {self.dataset_root}")
        print(f"Number of classes: {self.get_num_classes()}")
        
        print(f"\nClasses:")
        for i, name in enumerate(self.get_class_names()):
            print(f"  {i:2d}: {name}")
        
        stats = self.get_statistics()
        print(f"\nDataset splits:")
        for split, count in stats['splits'].items():
            print(f"  {split}: {count} images")
        
        print(f"\nTotal images: {stats['total_images']}")
        print("\n" + "="*70)


def create_road_sign_yaml(base_path: str, output_file: str = 'roadsigns.yaml'):
    """
    Create road sign dataset YAML file
    
    Args:
        base_path: Base path to dataset
        output_file: Output YAML filename
    """
    config = {
        'path': base_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 43,
        'names': [
            'Speed limit 20', 'Speed limit 30', 'Speed limit 50', 'Speed limit 60',
            'Speed limit 70', 'Speed limit 80', 'Speed limit 100', 'Speed limit 120',
            'No passing', 'No passing trucks', 'Right of way', 'Priority road',
            'Yield', 'Stop', 'No entry', 'No vehicles',
            'No horns', 'No parking', 'No stopping', 'Pedestrian crossing',
            'Children crossing', 'Bicycle crossing', 'Slippery road', 'Road work',
            'Traffic signals', 'Bumpy road', 'Uneven road', 'Road narrows',
            'Falling rocks', 'Animal crossing', 'End of speed limit',
            'Turn left', 'Turn right', 'Keep left', 'Keep right',
            'Roundabout', 'End no passing', 'End speed 80', 'End all restrictions',
            'Mandatory direction', 'One way', 'Pedestrian zone', 'Bicycle lane'
        ]
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created road sign YAML at {output_file}")
    return config


if __name__ == "__main__":
    # Example usage
    yaml_file = 'roadsigns.yaml'
    
    # Create YAML file
    create_road_sign_yaml('/path/to/dataset', yaml_file)
    
    # Load and validate
    dataset = RoadSignDataset(yaml_file)
    dataset.print_info()
    
    # Validate
    validation = dataset.validate_dataset()
    print(f"\nValidation results: {validation}")
