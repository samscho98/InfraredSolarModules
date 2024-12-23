# config.py
import os

# Data configuration
BASE_DIR = './data/InfraredSolarModules'
JSON_PATH = os.path.join(BASE_DIR, 'module_metadata.json')

# Model configuration
MODEL_INPUT_SHAPE = (224, 224, 3)
CHECKPOINT_DIR = 'model_checkpoints'
MODEL_SAVE_PATH = 'trained_model'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Training configuration
TRAIN_SIZE = 18000
TEST_SIZE = 2000
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 30

# Model architecture
CONV_LAYERS = [
    {'filters': 32, 'dropout': 0.25},
    {'filters': 64, 'dropout': 0.25},
    {'filters': 128, 'dropout': 0.25}
]
DENSE_LAYER_SIZE = 512
DENSE_DROPOUT = 0.5

# Training callbacks
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN = 1e-6

# Classes
ANOMALY_CLASSES = [
    'No-Anomaly', 'Cell', 'Hot-Spot', 'Offline-Module',
    'Vegetation', 'Diode', 'Shadowing', 'Cracking',
    'Diode-Multi', 'Hot-Spot-Multi', 'Cell-Multi', 'Soiling'
]

# Evaluation configuration
CONFUSION_MATRIX_FIGSIZE = (12, 10)
ROC_CURVE_FIGSIZE = (12, 8)