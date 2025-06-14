import os
import yaml
from ultralytics import YOLO
from tqdm import tqdm
import argparse

def create_dataset_yaml(data_path):
    """
    Create a YAML configuration file for the dataset based on existing structure
    """
    # Check if data.yaml already exists
    yaml_path = os.path.join(data_path, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"Using existing data.yaml configuration file at {yaml_path}")
        return yaml_path
    
    # Create the YAML file if it doesn't exist
    yaml_content = {
        'train': os.path.join(data_path, 'train', 'images'),
        'val': os.path.join(data_path, 'valid', 'images'),
        'test': os.path.join(data_path, 'test', 'images'),
        'nc': 2,  # Number of classes
        'names': ['Fire', 'Smoke']
    }
    
    # Save the YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {yaml_path}")
    return yaml_path

def train_model(data_yaml, epochs=5, batch_size=16, img_size=640, weights='yolov8m.pt'):
    """
    Train a YOLOv8 model for fire and smoke detection
    """
    # Initialize model
    model = YOLO(weights)
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='fire_smoke_detection',
        verbose=True,
        patience=2,  # Early stopping after 2 epochs with no improvement
        device='cpu',  # Use CPU since no GPU is available
        save_period=1,  # Save model after every epoch
        save=True,     # Save the final model
    )
    
    print(f"Training completed. Model saved at {os.path.join('runs/detect/fire_smoke_detection', 'weights/best.pt')}")
    print(f"Model checkpoints for each epoch saved in {os.path.join('runs/detect/fire_smoke_detection', 'weights/')}")
    return os.path.join('runs/detect/fire_smoke_detection', 'weights/best.pt')

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for fire and smoke detection')
    parser.add_argument('--data_path', type=str, default='DATASET', 
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--weights', type=str, default='yolov8m.pt',
                        help='Initial weights for training')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_path):
        print(f"Dataset directory {args.data_path} does not exist. Please create it with the proper structure.")
        return
    
    # Create or use existing dataset YAML file
    yaml_path = create_dataset_yaml(args.data_path)
    
    # Train the model
    best_model_path = train_model(
        yaml_path, 
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        weights=args.weights
    )
    
    print(f"Model training complete. Best model saved at: {best_model_path}")
    
if __name__ == "__main__":
    main() 