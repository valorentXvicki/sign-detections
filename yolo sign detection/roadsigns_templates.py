"""
Predefined road sign configurations and templates
"""

import yaml
from pathlib import Path
from typing import Dict, List

# European road signs (43 classes)
EUROPEAN_SIGNS = {
    'num_classes': 43,
    'signs': [
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

# German traffic signs (63 classes)
GERMAN_SIGNS = {
    'num_classes': 63,
    'signs': [
        # Speed limits (0-7)
        'Speed limit 20', 'Speed limit 30', 'Speed limit 40', 'Speed limit 50',
        'Speed limit 60', 'Speed limit 70', 'Speed limit 80', 'Speed limit 100',
        # Prohibition signs (8-15)
        'No entry', 'No vehicles', 'No motor vehicles', 'No trucks',
        'No buses', 'No cycles', 'No motorcycles', 'No horns',
        # Warning signs (16-31)
        'Danger ahead', 'Slippery road', 'Road work', 'Traffic signals',
        'Children crossing', 'Pedestrian crossing', 'Bicycle crossing',
        'Cattle crossing', 'Animal crossing', 'Ice/snow', 'Falling rocks',
        'Bumpy road', 'Uneven road', 'Road narrows', 'Steep hill',
        'Sharp left turn', 'Sharp right turn',
        # Mandatory signs (32-42)
        'Keep right', 'Keep left', 'Go straight', 'Turn left',
        'Turn right', 'Pass on left', 'Pass on right', 'Roundabout',
        'Mandatory right', 'Mandatory left', 'Pedestrian zone',
        # Information signs (43-62)
        'Parking', 'Hospital', 'Gas station', 'Restaurant',
        'Hotel', 'Campsite', 'Telephone', 'First aid',
        'Police station', 'Motorway', 'One way', 'Dead end',
        'Do not enter', 'Stop', 'Yield', 'No entry',
        'End of motorway', 'End of no passing', 'End of speed limit',
        'End prohibition'
    ]
}

# US road signs (simplified, commonly detected)
US_SIGNS = {
    'num_classes': 22,
    'signs': [
        'Speed limit', 'Yield', 'Stop', 'No entry',
        'Pedestrian crossing', 'Do not pass', 'No passing zone',
        'No left turn', 'No right turn', 'Keep right',
        'Keep left', 'Go straight', 'One way', 'Divided highway',
        'Warning', 'Construction', 'Dead end', 'No parking',
        'Disabled parking', 'Hospital', 'School zone', 'Playground'
    ]
}


def create_yaml_from_template(template: Dict, output_path: str, 
                             dataset_path: str = None) -> Dict:
    """
    Create YAML file from predefined template
    
    Args:
        template: Template dictionary with signs
        output_path: Path to save YAML file
        dataset_path: Path to dataset root
        
    Returns:
        Configuration dictionary
    """
    if dataset_path is None:
        dataset_path = '/path/to/dataset'
    
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': template['num_classes'],
        'names': template['signs']
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config


def get_template(region: str) -> Dict:
    """
    Get predefined template by region
    
    Args:
        region: 'european', 'german', or 'us'
        
    Returns:
        Template dictionary
    """
    templates = {
        'european': EUROPEAN_SIGNS,
        'german': GERMAN_SIGNS,
        'us': US_SIGNS
    }
    
    return templates.get(region, EUROPEAN_SIGNS)


def list_available_templates() -> List[str]:
    """List available templates"""
    return ['european', 'german', 'us']


def create_custom_yaml(signs: List[str], output_path: str,
                      dataset_path: str = None) -> Dict:
    """
    Create custom YAML file with user-defined signs
    
    Args:
        signs: List of sign names
        output_path: Path to save YAML file
        dataset_path: Path to dataset root
        
    Returns:
        Configuration dictionary
    """
    if dataset_path is None:
        dataset_path = '/path/to/dataset'
    
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(signs),
        'names': signs
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config


if __name__ == "__main__":
    print("Available templates:")
    for template in list_available_templates():
        print(f"  - {template}")
    
    print("\nCreating sample YAML files...")
    
    # European signs
    create_yaml_from_template(EUROPEAN_SIGNS, 'roadsigns_european.yaml')
    print("Created: roadsigns_european.yaml")
    
    # German signs
    create_yaml_from_template(GERMAN_SIGNS, 'roadsigns_german.yaml')
    print("Created: roadsigns_german.yaml")
    
    # US signs
    create_yaml_from_template(US_SIGNS, 'roadsigns_us.yaml')
    print("Created: roadsigns_us.yaml")
    
    # Custom signs
    custom_signs = ['Stop', 'Yield', 'Speed limit', 'No entry']
    create_custom_yaml(custom_signs, 'roadsigns_custom.yaml')
    print("Created: roadsigns_custom.yaml")
