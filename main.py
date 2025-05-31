import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import (Input, Dense, LSTM, TimeDistributed,
                                     Flatten, Dropout, GlobalAveragePooling1D) # Keep necessary layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import GlobalAveragePooling2D  # Import this layer
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import time
import glob
import itertools # Needed for generator chaining
print(os.listdir('/content/drive/MyDrive/Training'))
for folder in ["Abuse", "Assault", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting"]:
    folder_path = os.path.join('/content/drive/MyDrive/Training', folder)
    if os.path.exists(folder_path):
        print(f"{folder}: {os.listdir(folder_path)}")
# --- Configuration (Keep as before, adjust paths/parameters) ---
DATASET_PATH = '/content/drive/MyDrive/Training'
NORMAL_FOLDER = 'Training-Normal-Videos-Part-1'  # [cite: 2] ADJUST THIS
CRIME_FOLDERS = ["Abuse", "Assault", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting"] # [cite: 2] ADJUST THIS

IMG_HEIGHT = 128 # [cite: 2]
IMG_WIDTH = 128 # [cite: 2]
SEQUENCE_LENGTH = 20 # [cite: 2]
NUM_CLASSES = 1 # [cite: 3] Binary classification

PRETRAINED_MODEL = MobileNetV2 # [cite: 3]
FEATURE_EXTRACTOR_LAYER = 'out_relu' # [cite: 3]
LSTM_UNITS = 64 # [cite: 3]
DENSE_UNITS = 32 # [cite: 3]
DROPOUT_RATE = 0.4 # [cite: 3]

BATCH_SIZE = 8  # Try reducing batch size further if crashes persist
EPOCHS =2 # [cite: 3]
LEARNING_RATE = 0.0005 # [cite: 3]
AUTOTUNE = tf.data.AUTOTUNE # For tf.data pipeline

# --- Helper Functions ---

# 1. Generator to yield video paths and labels
def video_generator(dataset_path, crime_folders, normal_folder=None):
    """Yields (video_path, label) tuples."""
    # Crime videos (Label = 1)
    for crime_type in crime_folders:
        folder_path = os.path.join(dataset_path, crime_type)
        if os.path.isdir(folder_path):
            video_paths = glob.glob(os.path.join(folder_path, '*.mp4')) + \
                          glob.glob(os.path.join(folder_path, '*.avi'))
            for video_path in video_paths:
                yield video_path, 1

    # Normal videos (Label = 0), only if normal_folder is provided
    if normal_folder:
        normal_folder_path = os.path.join(dataset_path, normal_folder)
        if os.path.isdir(normal_folder_path):
            video_paths = glob.glob(os.path.join(normal_folder_path, '*.mp4')) + \
                          glob.glob(os.path.join(normal_folder_path, '*.avi'))
            for video_path in video_paths:
                yield video_path, 0
        else:
            print(f"Warning: Normal folder not found: {normal_folder_path}")

# 2. Function to load and process frames (for tf.data)
@tf.function # Decorator for potential graph mode optimization
def load_and_process_video(video_path, label, sequence_length, img_height, img_width):
    """Loads frames from a single video path using tf.py_function."""
    def _load_frames(path):
        path = path.numpy().decode('utf-8') # Decode tensor string
        sequences = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): # [cite: 3]
            print(f"Error: Could not open video {path}") # [cite: 4]
            # Return an empty array with the correct shape structure if error
            # Shape: (num_sequences, sequence_length, height, width, channels)
            return np.zeros((0, sequence_length, img_height, img_width, 3), dtype=np.float32)

        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (img_width, img_height)) # [cite: 5]
                normalized_frame = resized_frame / 255.0 # [cite: 5]
                frames.append(normalized_frame)

                if len(frames) == sequence_length:
                    sequences.append(np.array(frames)) # [cite: 5]
                    frames = frames[sequence_length // 2:] # Overlap [cite: 6]

        except Exception as e:
            print(f"Error processing video {path}: {e}") # [cite: 6]
             # Return empty on error during processing
            return np.zeros((0, sequence_length, img_height, img_width, 3), dtype=np.float32)
        finally:
            cap.release() # [cite: 6]

        if not sequences:
            # Handle videos shorter than sequence length or empty videos
             return np.zeros((0, sequence_length, img_height, img_width, 3), dtype=np.float32)

        return np.array(sequences, dtype=np.float32) # Ensure float32 for TF

    # Use tf.py_function to wrap the Python code
    sequences = tf.py_function(
        _load_frames,
        [video_path],
        tf.float32 # Output type
    )

    # Set shape information which is lost by py_function
    # Shape: (num_sequences, sequence_length, height, width, channels)
    sequences.set_shape([None, sequence_length, img_height, img_width, 3])

    # Repeat the label for each sequence extracted from the video
    num_sequences = tf.shape(sequences)[0]
    repeated_labels = tf.repeat(label, num_sequences)

    return sequences, repeated_labels

# --- Build Model (Keep the build_model function as it is) ---
# --- Build Model ---
def build_model(sequence_length, img_height, img_width, lstm_units, dense_units, dropout_rate,
                num_classes,
                learning_rate):
    """Builds the CNN + LSTM model."""
    input_shape = (sequence_length, img_height, img_width, 3)  #
    video_input = Input(shape=input_shape, name='video_input')

    # Load pre-trained CNN (without top classification layer)
    base_model = PRETRAINED_MODEL(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  #

    # --- Feature Extraction ---
    # Create a model that outputs the features from the chosen layer
    intermediate_model = Model(inputs=base_model.input,
                               outputs=base_model.get_layer(FEATURE_EXTRACTOR_LAYER).output,
                               # Use the corrected layer name
                               name="intermediate_feature_extractor")

    # Wrap the intermediate model AND a pooling layer with TimeDistributed
    # This applies the feature extractor and pooling to each frame (time step)
    time_distributed_features = TimeDistributed(
        keras.Sequential([  # Add a Sequential wrapper
            intermediate_model,
            GlobalAveragePooling2D()  # Add pooling here to flatten spatial dimensions
        ]),
        name='time_distributed_features'
    )(video_input)

    # --- Temporal Learning (LSTM) ---
    # Output of TimeDistributed(GlobalAveragePooling2D) should be (batch, seq_len, features)
    # which is suitable for LSTM
    lstm_out = LSTM(lstm_units, return_sequences=False, name='lstm_layer')(time_distributed_features)  #
    lstm_out = Dropout(dropout_rate)(lstm_out)  #

    # --- Classification Head ---
    x = Dense(dense_units, activation='relu', name='dense_1')(lstm_out)  #
    x = Dropout(dropout_rate)(x)  #
    output = Dense(num_classes, activation='sigmoid', name='output_layer')(x)  #

    # --- Compile Model ---
    model = Model(inputs=video_input, outputs=output, name='CrimePredictor')  #
    optimizer = Adam(learning_rate=learning_rate)  #
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  #
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),  #
                           tf.keras.metrics.Recall(name='recall')])  #

    print("Model Built Successfully:")
    model.summary(expand_nested=True)  # Use expand_nested to see inside the TimeDistributed layer
    return model


# --- Plotting History (Keep as it is) ---
def plot_training_history(history): # [cite: 23]
    # ... (Keep the exact plotting code [cite: 23, 24])
    pass # Replace with actual plotting code


# --- Main Execution ---
if __name__ == "__main__":
    print("TensorFlow Version:", tf.__version__) # [cite: 25]
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # [cite: 25]

    # 1. Create list of all video paths and labels
    print("--- Gathering Video Files ---")
    all_video_data = list(video_generator(DATASET_PATH, CRIME_FOLDERS, NORMAL_FOLDER if os.path.exists(os.path.join(DATASET_PATH, NORMAL_FOLDER)) else None))
    if not all_video_data:
      print("Error: No video files found. Check dataset paths and permissions.")
      exit()

    all_video_paths = [item[0] for item in all_video_data]
    all_labels = [item[1] for item in all_video_data]

    print(f"Found {len(all_video_paths)} total videos.")
    print(f"Class Distribution: Normal (0): {all_labels.count(0)}, Crime (1): {all_labels.count(1)}")

# Split file paths and labels
    print("\n--- Splitting Data (File Paths) ---")
    try:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_video_paths, all_labels,
        test_size=0.25, random_state=42, stratify=all_labels
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=0.20, random_state=42, stratify=train_labels
        )
    except ValueError as e:
        print(f"Error during splitting: {e}")
        print("Check if all classes have enough videos.")
        exit()

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")
    del all_video_data, all_video_paths, all_labels # Clear memory


    # 3. Create tf.data Datasets
    print("\n--- Creating Data Generators ---")

    # Training Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths)) # Shuffle paths before loading
    # Use flat_map to handle videos producing multiple sequences or no sequences
    train_ds = train_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
        load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH))
    )
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # Prefetch for performance
    # Optional: Caching after first epoch if dataset fits in memory (unlikely here)
    # train_ds = train_ds.cache()

    # Validation Dataset (similar pipeline, no shuffling within epochs)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
        load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH))
    )
    val_ds = val_ds.batch(BATCH_SIZE) # Use same batch size or larger if memory allows
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache() # Optional caching

     # Test Dataset (Needed for final evaluation)
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
        load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH))
    )
    test_ds = test_ds.batch(BATCH_SIZE) # Can use a larger batch size for evaluation
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


    # 4. Build Model
    print("\n--- Building Model ---")
    model = build_model(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
                        LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE,
                        NUM_CLASSES, LEARNING_RATE) # [cite: 28]

    # 5. Train Model using the datasets
    print("\n--- Starting Training ---")
    checkpoint_path = "crime_predictor_best_generator.keras" # [cite: 28] New checkpoint name
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                       save_best_only=True, # [cite: 29]
                                       monitor='val_accuracy', # [cite: 29]
                                       mode='max', # [cite: 30]
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', # [cite: 30]
                                   patience=10, # [cite: 30]
                                   mode='min', # [cite: 31]
                                   restore_best_weights=True, # [cite: 31]
                                   verbose=1) # [cite: 32]

    # Calculate steps per epoch if needed (useful for progress bars)
    # steps_per_epoch = len(train_paths) // BATCH_SIZE # Approximate, as #sequences varies
    # validation_steps = len(val_paths) // BATCH_SIZE # Approximate

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1
        # steps_per_epoch=steps_per_epoch, # Optional
        # validation_steps=validation_steps  # Optional
    ) # [cite: 32]

    print("\n--- Training Complete ---")
    try:
        plot_training_history(history) # [cite: 32]
    except Exception as plot_err:
        print(f"Could not plot training history: {plot_err}")


    # 6. Evaluate Model
    print("\n--- Evaluating on Test Set using Generator ---")
    print(f"Loading best model from: {checkpoint_path}")
    try:
        best_model = keras.models.load_model(checkpoint_path) # [cite: 33]
        # Evaluate using the test dataset generator
        loss, accuracy, precision, recall = best_model.evaluate(test_ds, verbose=1) # [cite: 33]
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}") # [cite: 34]

        # Detailed Classification Report (requires iterating through test_ds)
        print("\nGenerating Classification Report (may take time)...")
        y_true = []
        y_pred_prob = []
        for batch_x, batch_y in test_ds:
             y_true.extend(batch_y.numpy())
             batch_pred = best_model.predict(batch_x, verbose=0)
             y_pred_prob.extend(batch_pred.flatten())

        y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
        y_true = np.array(y_true)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Crime (1)'])) # [cite: 34]

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred)) # [cite: 34]

    except Exception as e:
        print(f"Error during evaluation with best model: {e}") # [cite: 35]
        # Optionally, try evaluating with the model state at the end of training
        # loss, accuracy, precision, recall = model.evaluate(test_ds, verbose=1)
        # print(f"Test Loss (End of Training): {loss:.4f}") ... etc. [cite: 35, 36]

    # 7. Example Prediction (Adapt load_video_frames for single prediction)
    print("\n--- Example Prediction ---")
    def load_single_video_for_prediction(video_path, sequence_length, img_height, img_width):
         # This is similar to the _load_frames in load_and_process_video
         # but returns directly, not wrapped in tf.py_function
        sequences = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return []
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                resized_frame = cv2.resize(frame, (img_width, img_height))
                normalized_frame = resized_frame / 255.0
                frames.append(normalized_frame)
                if len(frames) == sequence_length:
                    sequences.append(np.array(frames))
                    frames = frames[sequence_length // 2:]
        finally:
            cap.release()
        return np.array(sequences, dtype=np.float32) if sequences else None


    test_video_path = "/content/drive/MyDrive/RoadAccidents105_x264.mp4" # [cite: 37] Keep your test path
    if os.path.exists(test_video_path): # [cite: 37]
        test_sequences_np = load_single_video_for_prediction(test_video_path, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)
        if test_sequences_np is not None and len(test_sequences_np) > 0: # [cite: 37]
            print(f"Predicting on video: {test_video_path} ({len(test_sequences_np)} sequences)")
            # Use the loaded best_model
            predictions = best_model.predict(test_sequences_np) # [cite: 37]
            avg_prediction = np.mean(predictions) # [cite: 38]
            final_label = "Crime" if avg_prediction > 0.5 else "Normal" # [cite: 38]
            print(f"Average Predicted Probability: {avg_prediction:.4f}") # [cite: 38]
            print(f"Final Prediction for the video: {final_label}") # [cite: 38]
        else:
            print(f"Could not extract valid sequences from {test_video_path}") # [cite: 39]
    else:
        print(f"Test video not found: {test_video_path}") # [cite: 39]

    print("\n--- Script Finished ---") # [cite: 39]