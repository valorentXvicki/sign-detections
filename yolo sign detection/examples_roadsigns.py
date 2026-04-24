"""
Examples for road sign YAML dataset management
"""

from roadsigns_utils import RoadSignDataset
from roadsigns_templates import (
    create_yaml_from_template, get_template, 
    create_custom_yaml, list_available_templates
)
import os


def example_1_create_from_template():
    """Example 1: Create YAML from predefined template"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Create YAML from Template")
    print("="*60)
    
    # Available templates
    templates = list_available_templates()
    print(f"Available templates: {templates}")
    
    # Get European template
    template = get_template('european')
    print(f"\nEuropean signs: {template['num_classes']} classes")
    print(f"Classes: {template['signs'][:5]}...")
    
    # Create YAML
    create_yaml_from_template(
        template,
        'roadsigns_example.yaml',
        dataset_path='/dataset/roadsigns'
    )
    print("Created roadsigns_example.yaml")


def example_2_load_and_validate():
    """Example 2: Load and validate dataset YAML"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Load and Validate Dataset")
    print("="*60)
    
    # Create a sample YAML first
    yaml_file = 'roadsigns_sample.yaml'
    template = get_template('european')
    create_yaml_from_template(template, yaml_file)
    
    # Load dataset
    if os.path.exists(yaml_file):
        dataset = RoadSignDataset(yaml_file)
        
        # Print info
        dataset.print_info()
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"\nDataset statistics: {stats}")
        
        # Validate
        validation = dataset.validate_dataset()
        print(f"\nValidation results:")
        print(f"  Valid: {validation['valid']}")
        print(f"  Errors: {validation['errors']}")
        print(f"  Warnings: {validation['warnings']}")


def example_3_create_custom_yaml():
    """Example 3: Create custom YAML with specific signs"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Create Custom YAML")
    print("="*60)
    
    # Define custom signs
    custom_signs = [
        'Stop', 'Yield', 'Speed limit 30', 'Speed limit 50',
        'Speed limit 80', 'Speed limit 120', 'No entry',
        'No parking', 'Pedestrian crossing', 'School zone'
    ]
    
    print(f"Creating YAML with {len(custom_signs)} custom signs:")
    for i, sign in enumerate(custom_signs):
        print(f"  {i}: {sign}")
    
    # Create YAML
    create_custom_yaml(
        custom_signs,
        'roadsigns_custom.yaml',
        dataset_path='/my/dataset'
    )
    print("\nCreated roadsigns_custom.yaml")


def example_4_access_dataset_info():
    """Example 4: Access dataset information"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Access Dataset Information")
    print("="*60)
    
    yaml_file = 'roadsigns_european.yaml'
    template = get_template('european')
    create_yaml_from_template(template, yaml_file)
    
    if os.path.exists(yaml_file):
        dataset = RoadSignDataset(yaml_file)
        
        # Get specific information
        print(f"Total classes: {dataset.get_num_classes()}")
        
        # Get class name by ID
        print(f"\nClass 0: {dataset.get_class_name(0)}")
        print(f"Class 13: {dataset.get_class_name(13)}")
        
        # Get class ID by name
        print(f"\nClass ID of 'Stop': {dataset.get_class_id('Stop')}")
        print(f"Class ID of 'Yield': {dataset.get_class_id('Yield')}")
        
        # List all classes
        print("\nAll classes:")
        for i, name in enumerate(dataset.get_class_names()):
            print(f"  {i:2d}: {name}")


def example_5_export_statistics():
    """Example 5: Export dataset statistics"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Export Statistics")
    print("="*60)
    
    yaml_file = 'roadsigns_stats.yaml'
    template = get_template('german')
    create_yaml_from_template(template, yaml_file)
    
    if os.path.exists(yaml_file):
        dataset = RoadSignDataset(yaml_file)
        
        # Export statistics
        output_file = 'dataset_stats.json'
        dataset.export_stats(output_file)
        
        print(f"Exported statistics to {output_file}")
        
        # Print stats
        stats = dataset.get_statistics()
        print(f"\nDataset statistics:")
        print(f"  Number of classes: {stats['num_classes']}")
        print(f"  Total images: {stats['total_images']}")


def example_6_compare_templates():
    """Example 6: Compare different templates"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Compare Templates")
    print("="*60)
    
    templates = list_available_templates()
    
    print(f"Comparing {len(templates)} templates:\n")
    
    for template_name in templates:
        template = get_template(template_name)
        print(f"{template_name.upper()}:")
        print(f"  Classes: {template['num_classes']}")
        print(f"  Signs: {template['signs'][:3]}...")
        print()


if __name__ == "__main__":
    print("\nRoad Sign YAML Dataset Examples\n")
    print("Available examples:")
    print("1. example_1_create_from_template() - Create from template")
    print("2. example_2_load_and_validate() - Load and validate")
    print("3. example_3_create_custom_yaml() - Create custom YAML")
    print("4. example_4_access_dataset_info() - Access dataset info")
    print("5. example_5_export_statistics() - Export statistics")
    print("6. example_6_compare_templates() - Compare templates")
    
    print("\nRun examples:")
    print("python examples_roadsigns.py")
