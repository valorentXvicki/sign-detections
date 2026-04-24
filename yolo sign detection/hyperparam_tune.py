import itertools
import json
import os
from ultralytics import YOLO

def hyperparameter_search(
    data='data.yaml',
    model_size='n',
    param_grid=None,
    device=0,
    max_trials=10,
    results_file='hyperparam_results.json'
):
    """
    Simple grid search for YOLO hyperparameters.
    param_grid: dict with keys as param names and values as lists of possible values
    """
    if param_grid is None:
        param_grid = {
            'epochs': [50, 100],
            'imgsz': [416, 640],
            'batch': [8, 16],
            'lr0': [0.01, 0.001],
        }
    
    keys, values = zip(*param_grid.items())
    trials = list(itertools.product(*values))
    trials = trials[:max_trials]  # Limit number of trials
    results = []
    
    for i, trial in enumerate(trials):
        params = dict(zip(keys, trial))
        print(f"\nTrial {i+1}/{len(trials)}: {params}")
        model = YOLO(f'yolov8{model_size}.pt')
        res = model.train(
            data=data,
            epochs=params['epochs'],
            imgsz=params['imgsz'],
            batch=params['batch'],
            lr0=params['lr0'],
            device=device,
            project='runs/hyperparam',
            name=f"trial_{i+1}",
            save=False,
            verbose=False
        )
        best_fitness = res.get('best_fitness', None)
        results.append({'params': params, 'fitness': best_fitness})
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\nHyperparameter search complete. Results saved to {results_file}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Hyperparameter Tuning')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset YAML file')
    parser.add_argument('--model', type=str, default='n', help='YOLO model size: n/s/m/l/x')
    parser.add_argument('--epochs', type=str, default='50,100', help='Epochs (comma-separated)')
    parser.add_argument('--imgsz', type=str, default='416,640', help='Image sizes (comma-separated)')
    parser.add_argument('--batch', type=str, default='8,16', help='Batch sizes (comma-separated)')
    parser.add_argument('--lr0', type=str, default='0.01,0.001', help='Learning rates (comma-separated)')
    parser.add_argument('--device', type=int, default=0, help='Device (GPU index or -1 for CPU)')
    parser.add_argument('--max_trials', type=int, default=4, help='Max number of trials')
    parser.add_argument('--results_file', type=str, default='hyperparam_results.json', help='Results output file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    param_grid = {
        'epochs': [int(x) for x in args.epochs.split(',')],
        'imgsz': [int(x) for x in args.imgsz.split(',')],
        'batch': [int(x) for x in args.batch.split(',')],
        'lr0': [float(x) for x in args.lr0.split(',')],
    }
    hyperparameter_search(
        data=args.data,
        model_size=args.model,
        param_grid=param_grid,
        device=args.device,
        max_trials=args.max_trials,
        results_file=args.results_file
    )
