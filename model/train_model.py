import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Custom classification layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(33, activation='softmax')  # Change to 33 for 33 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dataset paths and generators
dataset_path = "data"  # Update this to point to your data folder
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class
    subset='validation'
)

# Print the number of classes detected
print("Number of classes:", train_generator.num_classes)

# Save the best model
checkpoint = ModelCheckpoint(
    'best_model_transfer_learning.keras', monitor='val_accuracy', save_best_only=True
)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=7,
    callbacks=[checkpoint]
)

# Save evaluation metrics
val_loss, val_accuracy = model.evaluate(validation_generator)
with open("model/metrics.txt", "w") as f:
    f.write(f"val_accuracy={val_accuracy}\n")
    f.write(f"val_loss={val_loss}\n")
print("Class indices:", train_generator.class_indices)