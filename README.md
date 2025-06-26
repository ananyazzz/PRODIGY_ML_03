# PRODIGY_ML_03
Load data
cat_images, cat_labels = load_images(train_dir, label_filter='cat') dog_images, dog_labels = load_images(train_dir, label_filter='dog')

X = np.concatenate((cat_images, dog_images), axis=0) y = np.concatenate((cat_labels, dog_labels), axis=0)

Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

Initialize the model
model = Sequential()

Convolutional Layer 1
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(MaxPool2D(pool_size=(2, 2))) model.add(Dropout(0.25))

Convolutional Layer 2
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(MaxPool2D(pool_size=(2, 2))) model.add(Dropout(0.25))

Convolutional Layer 3
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu', padding='same')) model.add(BatchNormalization()) model.add(MaxPool2D(pool_size=(2, 2))) model.add(Dropout(0.25))

Flatten and Dense Layers
model.add(Flatten()) model.add(Dense(128, activation='relu')) model.add(Dropout(0.25)) model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification

Compile the model
METRICS = [ 'accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall') ] model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=METRICS)

Model summary
model.summary()

Data augmentation
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True) train_generator = data_generator.flow(X_train, y_train, batch_size=batch_size) steps_per_epoch = X_train.shape[0] // batch_size

Fit the model
r = model.fit( train_generator, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=(X_val, y_val) )

Evaluate the model
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val) print(f"Validation Accuracy: {val_accuracy:.4f}") print(f"Validation Precision: {val_precision:.4f}") print(f"Validation Recall: {val_recall:.4f}")

Save the model architecture to JSON
import json import pickle

model_json = model.to_json() with open("model_architecture.json", "w") as json_file: json.dump(model_json, json_file)

Save the model weights to HDF5
model.save_weights("model_weights.h5")

Use pickle to save the paths to the architecture and weights files
model_paths = { "architecture": "model_architecture.json", "weights": "model_weights.h5" }

with open("model_paths.pkl", "wb") as f: pickle.dump(model_paths, f)

print("Model saved using pickle")
