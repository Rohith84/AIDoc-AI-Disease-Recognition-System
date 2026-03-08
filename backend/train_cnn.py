import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import glob
from PIL import ImageFile

# Fix for truncated images in datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_cnn_model(num_classes):
    """Create a CNN model for medical image classification"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def find_train_val_dirs(category_path):
    """Auto-detect train and validation directories for a category"""
    train_dir = None
    val_dir = None
    
    subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    
    # Check for nested structure (e.g., Bone_Fracture_Binary_Classification/)
    if len(subdirs) == 1 and not any(name.lower() in ['train', 'training', 'val', 'validation', 'test', 'testing'] for name in subdirs):
        # Likely a nested folder, go one level deeper
        nested_path = os.path.join(category_path, subdirs[0])
        subdirs = [d for d in os.listdir(nested_path) if os.path.isdir(os.path.join(nested_path, d))]
        category_path = nested_path
    
    for d in subdirs:
        d_lower = d.lower()
        if d_lower in ['train', 'training']:
            train_dir = os.path.join(category_path, d)
        elif d_lower in ['val', 'validation', 'test', 'testing']:
            if val_dir is None:  # Prefer val over test
                val_dir = os.path.join(category_path, d)
    
    return train_dir, val_dir


def train_single_model(category_name, category_path):
    """Train a CNN model for a single image category"""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {category_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    
    train_dir, val_dir = find_train_val_dirs(category_path)
    
    if not train_dir or not os.path.exists(train_dir):
        print(f"  [ERROR] No training directory found in {category_path}")
        return False
    
    if not val_dir or not os.path.exists(val_dir):
        print(f"  [WARNING] No validation directory found, using training data split")
        val_dir = None
    
    # Count classes
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"  [INFO] Found {len(classes)} classes: {classes}")
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        num_images = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        print(f"    - {cls}: {num_images} images")
    
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2 if val_dir is None else 0.0
    )
    
    # Validation data generator (no augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    # Training generator
    if val_dir is None:
        # Split training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
    
    # Create and train model
    num_classes = len(train_generator.class_indices)
    model = create_cnn_model(num_classes)
    
    print(f"\n  [TRAINING] Starting training with {num_classes} classes...")
    print(f"  [TRAINING] Training samples: {train_generator.samples}")
    
    if validation_generator:
        print(f"  [TRAINING] Validation samples: {validation_generator.samples}")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Calculate class weights to handle dataset imbalance
    from sklearn.utils import class_weight
    import numpy as np
    
    classes_list = train_generator.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes_list),
        y=classes_list
    )
    class_weights = dict(enumerate(weights))
    print(f"    - Computed class weights: {class_weights}")
    
    # Train
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Get final accuracy
    val_acc = max(history.history.get('val_accuracy', [0]))
    train_acc = max(history.history.get('accuracy', [0]))
    
    # Save model
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = f'cnn_{category_name}.h5'
    labels_filename = f'labels_{category_name}.json'
    
    model_path = os.path.join(backend_dir, model_filename)
    labels_path = os.path.join(backend_dir, labels_filename)
    
    model.save(model_path)
    
    with open(labels_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=2)
    
    print(f"\n  [OK] Model saved: {model_filename}")
    print(f"  [OK] Labels saved: {labels_filename}")
    print(f"  [OK] Training Accuracy: {train_acc*100:.2f}%")
    print(f"  [OK] Validation Accuracy: {val_acc*100:.2f}%")
    print(f"  [OK] Classes: {train_generator.class_indices}")
    
    return True


def train_all_models():
    """Auto-detect all image categories and train a model for each"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "..", "data", "images")
    
    if not os.path.exists(images_dir):
        print("[ERROR] data/images/ directory not found!")
        return
    
    # Find all category folders
    categories = [d for d in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, d))]
    
    if not categories:
        print("[ERROR] No image category folders found in data/images/")
        return
    
    print(f"\n{'='*60}")
    print(f"  AI DISEASE RECOGNITION - CNN TRAINING")
    print(f"{'='*60}")
    print(f"\n  Found {len(categories)} image categories: {categories}")
    print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU Device: {gpus[0].name}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    results = {}
    
    for category in sorted(categories):
        category_path = os.path.join(images_dir, category)
        success = train_single_model(category, category_path)
        results[category] = "✅ Success" if success else "❌ Failed"
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"  TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    for category, status in results.items():
        print(f"  {category}: {status}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    train_all_models()