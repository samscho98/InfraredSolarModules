import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import config as cfg

def load_data(base_dir=cfg.BASE_DIR, json_path=cfg.JSON_PATH):
    """
    Load images and their corresponding labels from JSON file
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        labels_data = json.load(f)
    
    images = []
    labels = []
    
    # Create label encoder for the anomaly classes
    label_encoder = LabelEncoder()
    label_encoder.fit(cfg.ANOMALY_CLASSES)
    
    # Load images and corresponding labels
    for img_id, data in labels_data.items():
        try:
            # Get image path from JSON
            img_path = os.path.join(base_dir, data['image_filepath'])
            
            # Load and preprocess image
            img = Image.open(img_path)
            img = img.resize(cfg.MODEL_INPUT_SHAPE[:2])
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
                img_array = np.repeat(img_array, 3, axis=-1)
            
            img_array = img_array.astype(np.float32) / 255.0
            
            images.append(img_array)
            labels.append(data['anomaly_class'])
            
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
    
    encoded_labels = label_encoder.transform(labels)
    categorical_labels = keras.utils.to_categorical(encoded_labels)
    
    return np.array(images), categorical_labels, label_encoder

def create_model(input_shape=cfg.MODEL_INPUT_SHAPE, num_classes=len(cfg.ANOMALY_CLASSES)):
    """
    Create a CNN model for anomaly classification
    """
    model = models.Sequential()
    
    for i, conv_config in enumerate(cfg.CONV_LAYERS):
        if i == 0:
            model.add(layers.Conv2D(conv_config['filters'], (3, 3), 
                                  activation='relu', padding='same', 
                                  input_shape=input_shape))
        else:
            model.add(layers.Conv2D(conv_config['filters'], (3, 3), 
                                  activation='relu', padding='same'))
        
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(conv_config['filters'], (3, 3), 
                              activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(conv_config['dropout']))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(cfg.DENSE_LAYER_SIZE, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(cfg.DENSE_DROPOUT))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def train_and_evaluate():
    """
    Train the model and evaluate its performance
    """
    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)
    
    print("Loading data...")
    images, labels, label_encoder = load_data()
    
    train_images = images[:cfg.TRAIN_SIZE]
    train_labels = labels[:cfg.TRAIN_SIZE]
    test_images = images[cfg.TRAIN_SIZE:cfg.TRAIN_SIZE + cfg.TEST_SIZE]
    test_labels = labels[cfg.TRAIN_SIZE:cfg.TRAIN_SIZE + cfg.TEST_SIZE]
    
    print("Creating model...")
    model = create_model()
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=cfg.REDUCE_LR_MIN
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(cfg.CHECKPOINT_DIR, 'model_{epoch:02d}-{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]
    
    print("Training model...")
    history = model.fit(
        train_images, train_labels,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=cfg.VALIDATION_SPLIT,
        callbacks=callbacks
    )
    
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    model.save(cfg.MODEL_SAVE_PATH)
    print(f"\nModel saved to {cfg.MODEL_SAVE_PATH}")
    
    with open(cfg.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {cfg.LABEL_ENCODER_PATH}")
    
    return model, label_encoder, history

def find_best_checkpoint(checkpoint_dir=cfg.CHECKPOINT_DIR):
    """
    Find the checkpoint with the lowest validation loss
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    
    checkpoints = []
    for filename in checkpoint_files:
        try:
            parts = filename.replace('.h5', '').split('-')
            epoch = int(parts[0].split('_')[1])
            val_loss = float(parts[1])
            
            checkpoints.append({
                'filename': filename,
                'epoch': epoch,
                'val_loss': val_loss,
                'full_path': os.path.join(checkpoint_dir, filename)
            })
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            continue
    
    df = pd.DataFrame(checkpoints)
    
    if df.empty:
        raise ValueError("No valid checkpoint files found")
    
    df = df.sort_values('val_loss')
    best_checkpoint = df.iloc[0]
    
    print("\nCheckpoint Summary:")
    print(df.to_string(index=False))
    print(f"\nBest Checkpoint:")
    print(f"Filename: {best_checkpoint['filename']}")
    print(f"Epoch: {best_checkpoint['epoch']}")
    print(f"Validation Loss: {best_checkpoint['val_loss']:.4f}")
    
    return best_checkpoint['full_path']

def load_best_model(checkpoint_dir=cfg.CHECKPOINT_DIR):
    """
    Find and load the best checkpoint model
    """
    best_checkpoint_path = find_best_checkpoint(checkpoint_dir)
    print(f"\nLoading model from {best_checkpoint_path}")
    model = keras.models.load_model(best_checkpoint_path)
    return model

def evaluate_existing_model(model_path=cfg.MODEL_SAVE_PATH, data_dir=None, json_path=None):
    """
    Evaluate an existing trained model on test data
    """
    print("Loading saved model...")
    model = keras.models.load_model(model_path)
    
    with open(cfg.LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    if data_dir is None:
        data_dir = cfg.BASE_DIR
    if json_path is None:
        json_path = cfg.JSON_PATH
    
    print("Loading test data...")
    images, labels, _ = load_data(data_dir, json_path)
    
    test_images = images[cfg.TRAIN_SIZE:cfg.TRAIN_SIZE + cfg.TEST_SIZE]
    test_labels = labels[cfg.TRAIN_SIZE:cfg.TRAIN_SIZE + cfg.TEST_SIZE]
    
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=label_encoder.classes_
    ))
    
    return test_accuracy, report

def predict_single_image(image_path, model_path=cfg.MODEL_SAVE_PATH):
    """
    Predict anomaly class for a single image
    
    Parameters:
    image_path (str): Path to the image file
    model_path (str): Path to the model to use for prediction
    
    Returns:
    tuple: (predicted_class, confidence_score)
    """
    try:
        # Load the model
        model = keras.models.load_model(model_path)
        
        # Load label encoder
        with open(cfg.LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(cfg.MODEL_INPUT_SHAPE[:2])
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure we have a 3D array (height, width, channels)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        # Get class name
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get confidence scores for all classes
        class_scores = {
            class_name: float(score) 
            for class_name, score in zip(label_encoder.classes_, prediction[0])
        }
        
        # Sort classes by confidence
        sorted_classes = sorted(
            class_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print results
        print(f"\nPrediction for image: {os.path.basename(image_path)}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print("\nTop 3 predictions:")
        for class_name, score in sorted_classes[:3]:
            print(f"{class_name}: {score:.2%}")
        
        return predicted_class, confidence, class_scores
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None