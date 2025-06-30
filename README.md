# Chest X-ray Detection Challenge

This repository contains code for chest X-ray abnormality detection using deep learning object detection models. The project implements and compares two approaches: YOLOv8 and Faster R-CNN.

## Project Overview

The goal of this project is to detect various chest X-ray abnormalities (e.g., Effusion, Infiltration, Mass, Nodule, Pneumothorax, etc.) using deep learning object detection models. The project includes:

- Data exploration and preprocessing
- Implementation of two detection models (YOLOv8 and Faster R-CNN)
- Data augmentation techniques for handling imbalanced classes
- Model training, validation, and testing
- Prediction generation for submission

## Directory Structure

```
├── data/                   # Data files
│   ├── ID_to_Image_Mapping.csv
│   ├── train.csv
├── src/                    # Source code
│   ├── data_augmentation.py  # Data preprocessing and augmentation
│   ├── model_FasterRCNN.py   # Faster R-CNN implementation
│   ├── model_YOLO.ipynb      # YOLOv8 implementation
│   ├── predict.py            # Prediction script
├── test/                   # Test images
├── train/                  # Training images
├── outputs/                # Model outputs
├── runs/                   # Training runs
├── EDA.ipynb               # Exploratory data analysis
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Lock file for dependencies
├── submission.csv          # Final predictions for submission
```

## Environment Setup

This project uses `uv` for dependency management. Follow these steps to set up the environment:

1. Install `uv` if you don't have it already:
   ```bash
   curl -sSf https://install.ultraviolet.rs | sh
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Chest_X-ray_Detection_Challenge
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   ```

5. Install YOLOv8 (if not included in dependencies):
   ```bash
   uv pip install ultralytics
   ```

## Data Preparation

1. Place your training images in the `train/` directory
2. Place your test images in the `test/` directory
3. Ensure `data/train.csv` contains the annotations in the format:
   - `Image Index`: filename
   - `Finding Label`: abnormality class
   - `Bbox [x`, `y`, `w`, `h]`: bounding box coordinates

## Model Training

### YOLOv8 Model

The YOLOv8 model training process is implemented in `src/model_YOLO.ipynb` and includes:

1. Data preparation and conversion to YOLO format
2. Class weight calculation to handle class imbalance
3. Rare class augmentation using a patch bank approach
4. Model training with carefully tuned hyperparameters
5. Two-phase training strategy with layer freezing

To train the model:
```bash
# Run the notebook cells in order
jupyter notebook src/model_YOLO.ipynb
```

### Faster R-CNN Model

The Faster R-CNN model is implemented in `src/model_FasterRCNN.py` and includes:

1. Custom dataset and data loader implementations
2. Multi-phase training with gradual unfreezing strategy
3. Extensive evaluation metrics (mAP, precision, recall, F1)

To train the Faster R-CNN model:
```bash
python src/model_FasterRCNN.py
```

## Making Predictions

1. For YOLOv8, run the prediction cells in the notebook
2. For Faster R-CNN, use:
   ```bash
   python src/predict.py
   ```

3. The final predictions will be saved in `submission.csv`

## Key Features

- **Data Augmentation**: Implemented custom augmentation for rare classes
- **Model Architecture**: Used pre-trained models with fine-tuning strategies
- **Evaluation Metrics**: Comprehensive evaluation using mAP, precision, recall, and F1 score
- **Rare Class Handling**: Special techniques to improve detection of rare abnormalities

## Performance

The models were evaluated using mean Average Precision (mAP) and other metrics. Detailed results can be found in the training logs under the `runs/` directory.

## Requirements

Key dependencies:
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- torchvision
- OpenCV
- pandas
- scikit-learn
- matplotlib
- tensorboard (for visualization)

All dependencies are specified in `pyproject.toml`.

## Notes

- The training process may require significant computational resources (GPU recommended)
- Pre-trained weights are used as starting points for both models
- Special attention is given to rare classes through custom augmentation techniques
- The COCO evaluation metric is used for consistent performance comparison
