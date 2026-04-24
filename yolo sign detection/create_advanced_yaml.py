"""
Enhanced road sign YAML with additional metadata
"""

import yaml
from typing import Dict
from pathlib import Path


def create_advanced_roadsigns_yaml(output_path: str = 'roadsigns_advanced.yaml'):
    """Create advanced YAML with metadata"""
    
    config = {
        'metadata': {
            'description': 'European Road Sign Detection Dataset',
            'version': '1.0',
            'author': 'Traffic Safety AI',
            'license': 'CC BY-NC-SA 4.0',
            'format': 'YOLO v8'
        },
        
        'dataset': {
            'path': '/path/to/road_signs_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test'
        },
        
        'classes': {
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
        },
        
        'class_info': {
            'warning': {
                'description': 'Yellow diamond warning signs',
                'color': 'yellow',
                'classes': [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            },
            'prohibition': {
                'description': 'Red circle prohibition signs',
                'color': 'red',
                'classes': [8, 9, 14, 15, 16, 17, 18]
            },
            'mandatory': {
                'description': 'Blue circle mandatory signs',
                'color': 'blue',
                'classes': [10, 11, 12, 13, 31, 32, 33, 34, 35, 39, 40, 41, 42]
            },
            'information': {
                'description': 'White/gray information signs',
                'color': 'white',
                'classes': [30, 36, 37, 38]
            }
        },
        
        'preprocessing': {
            'img_size': 640,
            'batch_size': 16,
            'augment': True,
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10,
                'translate': 0.1,
                'scale': 0.5,
                'flipud': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
        },
        
        'model': {
            'architecture': 'YOLOv8',
            'variants': ['nano', 'small', 'medium', 'large', 'extra-large'],
            'recommended': 'small'
        },
        
        'training': {
            'epochs': 100,
            'patience': 20,
            'device': 'cuda',
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937
        },
        
        'evaluation': {
            'metrics': ['mAP50', 'mAP75', 'mAP50-95'],
            'iou_threshold': 0.5,
            'conf_threshold': 0.5
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created advanced YAML at {output_path}")
    return config


def create_multilingual_roadsigns_yaml(output_path: str = 'roadsigns_multilingual.yaml'):
    """Create YAML with multilingual labels"""
    
    config = {
        'dataset': {
            'path': '/path/to/road_signs_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test'
        },
        
        'classes': {
            'nc': 43
        },
        
        'names': {
            'english': [
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
            ],
            'german': [
                'Geschwindigkeitsbegrenzung 20', 'Geschwindigkeitsbegrenzung 30',
                'Geschwindigkeitsbegrenzung 50', 'Geschwindigkeitsbegrenzung 60',
                'Geschwindigkeitsbegrenzung 70', 'Geschwindigkeitsbegrenzung 80',
                'Geschwindigkeitsbegrenzung 100', 'Geschwindigkeitsbegrenzung 120',
                'Überholverbot', 'Überholverbot für Lastkraftwagen',
                'Vorfahrtsrecht', 'Vorrangstraße',
                'Vorfahrt gewähren', 'Halt', 'Einfahrt verboten', 'Eintritt verboten',
                'Hupen verboten', 'Parken verboten', 'Halten verboten',
                'Fußgängerüberweg', 'Kinder', 'Fahrradverkehr',
                'Rutschgefahr', 'Baustelle', 'Ampel',
                'Bodenschwellen', 'Unebene Fahrbahn', 'Fahrbahnverengung',
                'Steinschlag', 'Wildwechsel', 'Ende der Geschwindigkeitsbegrenzung',
                'Links abbiegen', 'Rechts abbiegen', 'Links halten',
                'Rechts halten', 'Kreisverkehr', 'Ende des Überholverbots',
                'Ende der 80er-Begrenzung', 'Ende aller Beschränkungen',
                'Anordnungsrichtung', 'Einbahnstraße',
                'Fußgängerzone', 'Fahrradweg'
            ],
            'french': [
                'Limitation de vitesse 20', 'Limitation de vitesse 30',
                'Limitation de vitesse 50', 'Limitation de vitesse 60',
                'Limitation de vitesse 70', 'Limitation de vitesse 80',
                'Limitation de vitesse 100', 'Limitation de vitesse 120',
                'Dépassement interdit', 'Dépassement interdit pour camions',
                'Priorité à droite', 'Route prioritaire',
                'Cédez le passage', 'Arrêt', 'Entrée interdite', 'Entrée interdite',
                'Klaxon interdit', 'Stationnement interdit', 'Arrêt interdit',
                'Passage pour piétons', 'Enfants', 'Passage pour vélos',
                'Route glissante', 'Travaux', 'Feux tricolores',
                'Ralentisseurs', 'Route bosselée', 'Rétrécissement de la route',
                'Chutes de pierres', 'Traversée d\'animaux',
                'Fin de limitation de vitesse',
                'Tournez à gauche', 'Tournez à droite', 'Tenez votre gauche',
                'Tenez votre droite', 'Rond-point', 'Fin de l\'interdiction de dépasser',
                'Fin du 80', 'Fin de toutes les restrictions',
                'Direction obligatoire', 'Rue à sens unique',
                'Zone piétonne', 'Piste cyclable'
            ],
            'spanish': [
                'Límite de velocidad 20', 'Límite de velocidad 30',
                'Límite de velocidad 50', 'Límite de velocidad 60',
                'Límite de velocidad 70', 'Límite de velocidad 80',
                'Límite de velocidad 100', 'Límite de velocidad 120',
                'Prohibido adelantar', 'Prohibido adelantar camiones',
                'Preferencia', 'Carretera prioritaria',
                'Ceder el paso', 'Parada', 'Prohibida la entrada',
                'Vehículos prohibidos', 'Prohibidas las bocinas',
                'Prohibido estacionar', 'Prohibido parar',
                'Paso peatonal', 'Zona infantil', 'Paso ciclista',
                'Carretera resbaladiza', 'Obras', 'Semáforos',
                'Badén', 'Carretera con baches', 'Estrechamiento de calzada',
                'Caída de rocas', 'Cruce de animales', 'Fin del límite de velocidad',
                'Gire a la izquierda', 'Gire a la derecha',
                'Manténgase a la izquierda', 'Manténgase a la derecha',
                'Rotonda', 'Fin de prohibición de adelantar', 'Fin del límite de 80',
                'Fin de restricciones', 'Dirección obligatoria',
                'Calle de sentido único', 'Zona peatonal', 'Carril bici'
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created multilingual YAML at {output_path}")
    return config


if __name__ == "__main__":
    print("Creating advanced YAML files...\n")
    
    create_advanced_roadsigns_yaml()
    create_multilingual_roadsigns_yaml()
    
    print("\nDone!")
