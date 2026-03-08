"""
Train an image-type router model using transfer learning (MobileNetV2).

This model classifies medical images into categories (chest_xray, brain_tumor,
bone_fracture) so the correct specialist model can be used for diagnosis.

Uses ALL 3 datasets together so the model learns to distinguish between
different types of medical images.

Usage:
    python train_router.py
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import shutil
import random
from PIL import ImageFile

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_router_model(num_categories):
    """Create a router model using MobileNetV2 transfer learning."""
    # Use MobileNetV2 as base - lightweight but powerful
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model layers (use pre-trained features)
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_categories, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_router():
    """Train the image-type router using samples from each category's training data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "..", "data", "images")
    
    if not os.path.exists(images_dir):
        print("[ERROR] data/images/ directory not found!")
        return
    
    # Category mapping: find training images for each category
    category_train_dirs = {
        'bone_fracture': os.path.join(images_dir, 'bone_fracture', 'Bone_Fracture_Binary_Classification', 'train'),
        'brain_tumor': os.path.join(images_dir, 'brain_tumor', 'Training'),
        'chest_xray': os.path.join(images_dir, 'chest_xray', 'train'),
    }
    
    print("=" * 60)
    print("  IMAGE TYPE ROUTER - TRAINING (MobileNetV2)")
    print("=" * 60)
    
    # Build temp directory with images labeled by category (not by disease)
    router_data_dir = os.path.join(current_dir, '..', 'data', '_router_temp')
    
    # Clean up any previous temp data
    if os.path.exists(router_data_dir):
        shutil.rmtree(router_data_dir)
    
    MAX_PER_CATEGORY = 1000  # More samples for better accuracy
    
    for category, src_dir in category_train_dirs.items():
        if not os.path.exists(src_dir):
            print(f"[ERROR] Training dir not found for {category}: {src_dir}")
            return
        
        # Collect all images from all sub-classes
        all_images = []
        for sub_class in os.listdir(src_dir):
            sub_path = os.path.join(src_dir, sub_class)
            if os.path.isdir(sub_path):
                for img_file in os.listdir(sub_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        all_images.append(os.path.join(sub_path, img_file))
        
        print(f"  {category}: found {len(all_images)} total images")
        
        # Subsample if needed for balanced classes
        random.seed(42)
        if len(all_images) > MAX_PER_CATEGORY:
            all_images = random.sample(all_images, MAX_PER_CATEGORY)
        
        # Split 80/20 for train/val
        random.shuffle(all_images)
        split = int(0.8 * len(all_images))
        train_images = all_images[:split]
        val_images = all_images[split:]
        
        # Copy images to temp directory structure
        for subset, images in [('train', train_images), ('val', val_images)]:
            dest_dir = os.path.join(router_data_dir, subset, category)
            os.makedirs(dest_dir, exist_ok=True)
            for i, img_path in enumerate(images):
                ext = os.path.splitext(img_path)[1]
                dest_path = os.path.join(dest_dir, f"{category}_{i}{ext}")
                shutil.copy2(img_path, dest_path)
        
        print(f"    -> Using: train={len(train_images)}, val={len(val_images)}")
    
    train_dir = os.path.join(router_data_dir, 'train')
    val_dir = os.path.join(router_data_dir, 'val')
    
    # Data generators - use MobileNetV2 preprocessing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    num_categories = len(train_gen.class_indices)
    print(f"\n  [INFO] Categories: {train_gen.class_indices}")
    print(f"  [INFO] Training samples: {train_gen.samples}")
    print(f"  [INFO] Validation samples: {val_gen.samples}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  [INFO] GPU: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create and train model
    model = create_router_model(num_categories)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )
    
    print(f"\n  [TRAINING] Starting training...")
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early_stopping]
    )
    
    # Save model - use standard rescale (not mobilenet preprocess) for compatibility
    # We need to save it so model.py can load it with simple /255 normalization
    # Actually, let's save it as-is and update model.py to use mobilenet preprocessing
    model_path = os.path.join(current_dir, 'cnn_router.h5')
    labels_path = os.path.join(current_dir, 'labels_router.json')
    
    model.save(model_path)
    with open(labels_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    
    val_acc = max(history.history.get('val_accuracy', [0]))
    train_acc = max(history.history.get('accuracy', [0]))
    print(f"\n  [OK] Router model saved: cnn_router.h5")
    print(f"  [OK] Router labels saved: labels_router.json")
    print(f"  [OK] Training accuracy: {train_acc*100:.2f}%")
    print(f"  [OK] Validation accuracy: {val_acc*100:.2f}%")
    print(f"  [OK] Categories: {train_gen.class_indices}")
    
    # Cleanup temp directory
    shutil.rmtree(router_data_dir)
    print(f"  [OK] Cleaned up temp data")


if __name__ == '__main__':
    train_router()
