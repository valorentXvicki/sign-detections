"""
Road sign dataset structure and utility functions
"""

from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Road sign classes with descriptions
ROAD_SIGNS = {
    0: {'name': 'Speed limit 20', 'type': 'warning', 'color': 'red/white'},
    1: {'name': 'Speed limit 30', 'type': 'warning', 'color': 'red/white'},
    2: {'name': 'Speed limit 50', 'type': 'warning', 'color': 'red/white'},
    3: {'name': 'Speed limit 60', 'type': 'warning', 'color': 'red/white'},
    4: {'name': 'Speed limit 70', 'type': 'warning', 'color': 'red/white'},
    5: {'name': 'Speed limit 80', 'type': 'warning', 'color': 'red/white'},
    6: {'name': 'Speed limit 100', 'type': 'warning', 'color': 'red/white'},
    7: {'name': 'Speed limit 120', 'type': 'warning', 'color': 'red/white'},
    8: {'name': 'No passing', 'type': 'prohibition', 'color': 'red/white'},
    9: {'name': 'No passing trucks', 'type': 'prohibition', 'color': 'red/white'},
    10: {'name': 'Right of way', 'type': 'mandatory', 'color': 'yellow/red'},
    11: {'name': 'Priority road', 'type': 'mandatory', 'color': 'yellow'},
    12: {'name': 'Yield', 'type': 'mandatory', 'color': 'red/white'},
    13: {'name': 'Stop', 'type': 'mandatory', 'color': 'red/white'},
    14: {'name': 'No entry', 'type': 'prohibition', 'color': 'red/white'},
    15: {'name': 'No vehicles', 'type': 'prohibition', 'color': 'red'},
    16: {'name': 'No horns', 'type': 'prohibition', 'color': 'blue/red'},
    17: {'name': 'No parking', 'type': 'prohibition', 'color': 'red/blue'},
    18: {'name': 'No stopping', 'type': 'prohibition', 'color': 'red/blue'},
    19: {'name': 'Pedestrian crossing', 'type': 'warning', 'color': 'yellow'},
    20: {'name': 'Children crossing', 'type': 'warning', 'color': 'yellow'},
    21: {'name': 'Bicycle crossing', 'type': 'warning', 'color': 'yellow'},
    22: {'name': 'Slippery road', 'type': 'warning', 'color': 'yellow'},
    23: {'name': 'Road work', 'type': 'warning', 'color': 'yellow'},
    24: {'name': 'Traffic signals', 'type': 'warning', 'color': 'yellow'},
    25: {'name': 'Bumpy road', 'type': 'warning', 'color': 'yellow'},
    26: {'name': 'Uneven road', 'type': 'warning', 'color': 'yellow'},
    27: {'name': 'Road narrows', 'type': 'warning', 'color': 'yellow'},
    28: {'name': 'Falling rocks', 'type': 'warning', 'color': 'yellow'},
    29: {'name': 'Animal crossing', 'type': 'warning', 'color': 'yellow'},
    30: {'name': 'End of speed limit', 'type': 'information', 'color': 'white/gray'},
    31: {'name': 'Turn left', 'type': 'mandatory', 'color': 'blue'},
    32: {'name': 'Turn right', 'type': 'mandatory', 'color': 'blue'},
    33: {'name': 'Keep left', 'type': 'mandatory', 'color': 'blue'},
    34: {'name': 'Keep right', 'type': 'mandatory', 'color': 'blue'},
    35: {'name': 'Roundabout', 'type': 'mandatory', 'color': 'blue'},
    36: {'name': 'End no passing', 'type': 'information', 'color': 'white/gray'},
    37: {'name': 'End speed 80', 'type': 'information', 'color': 'white/gray'},
    38: {'name': 'End all restrictions', 'type': 'information', 'color': 'white/gray'},
    39: {'name': 'Mandatory direction', 'type': 'mandatory', 'color': 'blue'},
    40: {'name': 'One way', 'type': 'mandatory', 'color': 'blue'},
    41: {'name': 'Pedestrian zone', 'type': 'mandatory', 'color': 'blue'},
    42: {'name': 'Bicycle lane', 'type': 'mandatory', 'color': 'blue'},
}

# Sign types
SIGN_TYPES = {
    'warning': 'Yellow diamond signs',
    'prohibition': 'Red circle signs',
    'mandatory': 'Blue circle signs',
    'information': 'White/Gray rectangular signs'
}


def load_yaml(yaml_path: str) -> Dict:
    """
    Load YAML dataset file
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary with dataset configuration
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded dataset configuration from {yaml_path}")
    logger.info(f"Dataset has {config['nc']} classes")
    
    return config


def save_yaml(config: Dict, yaml_path: str):
    """
    Save YAML dataset file
    
    Args:
        config: Dataset configuration dictionary
        yaml_path: Path to save YAML file
    """
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved dataset configuration to {yaml_path}")


def create_dataset_structure(base_path: str):
    """
    Create dataset directory structure
    
    Args:
        base_path: Base path for dataset
    """
    base = Path(base_path)
    
    # Create directories
    dirs = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test',
        'annotations'
    ]
    
    for d in dirs:
        path = base / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def get_sign_info(class_id: int) -> Dict:
    """
    Get information about a specific road sign
    
    Args:
        class_id: Class ID of the sign
        
    Returns:
        Dictionary with sign information
    """
    if class_id not in ROAD_SIGNS:
        logger.warning(f"Unknown class ID: {class_id}")
        return None
    
    return ROAD_SIGNS[class_id]


def get_signs_by_type(sign_type: str) -> List[Tuple[int, str]]:
    """
    Get all signs of a specific type
    
    Args:
        sign_type: Type of sign ('warning', 'prohibition', 'mandatory', 'information')
        
    Returns:
        List of (class_id, sign_name) tuples
    """
    signs = []
    
    for class_id, info in ROAD_SIGNS.items():
        if info['type'] == sign_type:
            signs.append((class_id, info['name']))
    
    return sorted(signs)


def print_dataset_info():
    """Print dataset information"""
    print("\n" + "="*70)
    print("ROAD SIGN DATASET INFORMATION")
    print("="*70)
    
    print(f"\nTotal classes: {len(ROAD_SIGNS)}")
    
    # Group by type
    for sign_type, description in SIGN_TYPES.items():
        signs = get_signs_by_type(sign_type)
        print(f"\n{sign_type.upper()} ({description}): {len(signs)} signs")
        
        for class_id, name in signs:
            print(f"  {class_id:2d}: {name}")
    
    print("\n" + "="*70)


def create_sample_yaml(output_path: str = 'roadsigns.yaml'):
    """
    Create sample YAML file
    
    Args:
        output_path: Path to save YAML file
    """
    config = {
        'path': '/path/to/road_signs_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(ROAD_SIGNS),
        'names': [ROAD_SIGNS[i]['name'] for i in range(len(ROAD_SIGNS))]
    }
    
    save_yaml(config, output_path)
    logger.info(f"Created sample YAML at {output_path}")


if __name__ == "__main__":
    print_dataset_info()
    create_sample_yaml()
