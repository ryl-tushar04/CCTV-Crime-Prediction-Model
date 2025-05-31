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
# [source: 2] import matplotlib.pyplot as plt # Removed for Streamlit, unless needed for other plots
import time
import glob
import itertools # Needed for generator chaining
import tempfile # Needed for Streamlit file handling
import streamlit as st # Import Streamlit

# --- Configuration (Keep as before, adjust paths/parameters if needed) ---
# [source: 2] DATASET_PATH = '/content/drive/MyDrive/Training'
# [source: 2] NORMAL_FOLDER = 'Training-Normal-Videos-Part-1'  # ADJUST THIS
# [source: 2] CRIME_FOLDERS = ["Abuse", "Assault", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting"] # ADJUST THIS

IMG_HEIGHT = 128 # [cite: 2]
IMG_WIDTH = 128 # [cite: 2]
SEQUENCE_LENGTH = 20 # [cite: 2]
# [source: 3] NUM_CLASSES = 1 # Binary classification
PRETRAINED_MODEL = MobileNetV2 # [cite: 3]
FEATURE_EXTRACTOR_LAYER = 'out_relu' # [cite: 3]
LSTM_UNITS = 64 # [cite: 3]
DENSE_UNITS = 32 # [cite: 3]
DROPOUT_RATE = 0.4 # [cite: 3]
# [source: 3] BATCH_SIZE = 8
# [source: 3] EPOCHS = 2
# [source: 3] LEARNING_RATE = 0.0005
# [source: 3] AUTOTUNE = tf.data.AUTOTUNE # For tf.data pipeline

# --- Helper Functions (Keep as they are) ---

# 1. Generator to yield video paths and labels (Used for training, not Streamlit app)
def video_generator(dataset_path, crime_folders, normal_folder=None):
# [source: 4]     """Yields (video_path, label) tuples."""
    # Crime videos (Label = 1)
    for crime_type in crime_folders:
        folder_path = os.path.join(dataset_path, crime_type)
        if os.path.isdir(folder_path):
            video_paths = glob.glob(os.path.join(folder_path, '*.mp4')) + \
                          glob.glob(os.path.join(folder_path, '*.avi')) # [cite: 4]
            for video_path in video_paths:
                yield video_path, 1

    # Normal videos (Label = 0), only if normal_folder is provided
    if normal_folder:
        normal_folder_path = os.path.join(dataset_path, normal_folder)
        if os.path.isdir(normal_folder_path): # [cite: 5]
            video_paths = glob.glob(os.path.join(normal_folder_path, '*.mp4')) + \
                          glob.glob(os.path.join(normal_folder_path, '*.avi')) # [cite: 5]
            for video_path in video_paths:
                yield video_path, 0
# [source: 6]         else:
            print(f"Warning: Normal folder not found: {normal_folder_path}") # [cite: 6]

# 2. Function to load and process frames for tf.data (Used for training, not Streamlit app)
@tf.function # Decorator for potential graph mode optimization
def load_and_process_video(video_path, label, sequence_length, img_height, img_width):
# [source: 6]     """Loads frames from a single video path using tf.py_function."""
    def _load_frames(path):
        path = path.numpy().decode('utf-8') # Decode tensor string
        sequences = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): # [cite: 3, 6]
# [source: 7]             print(f"Error: Could not open video {path}") # [cite: 4, 7]
            # Return an empty array with the correct shape structure if error
            # Shape: (num_sequences, sequence_length, height, width, channels)
            return np.zeros((0, sequence_length, img_height, img_width, 3), dtype=np.float32) # [cite: 7]

        frames = []
        try:
            while True: # [cite: 8]
                ret, frame = cap.read() # [cite: 8]
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (img_width, img_height)) # [cite: 5, 8]
                normalized_frame = resized_frame / 255.0 # [cite: 5, 9]
                frames.append(normalized_frame) # [cite: 9]

                if len(frames) == sequence_length:
                    sequences.append(np.array(frames)) # [cite: 5, 9]
                    frames = frames[sequence_length // 2:] # Overlap [cite: 6, 9]

        except Exception as e:
            print(f"Error processing video {path_str}: {e}")
            # Return empty list to indicate failure inside py_function
            return np.array([], dtype=np.float32).reshape(0, sequence_length, img_height, img_width, 3) # Reshape
        finally:
            if cap is not None: # Check if cap was successfully opened
                cap.release()

        if not sequences:
             # Handle videos shorter than sequence length or empty videos
             # Return empty list to indicate no sequences
             return np.array([], dtype=np.float32).reshape(0, sequence_length, img_height, img_width, 3) # Reshape

        return np.array(sequences, dtype=np.float32) # Return the valid sequences
    # Use tf.py_function to wrap the Python code
    sequences = tf.py_function(
        _load_frames,
        [video_path],
        tf.float32 # Output type [cite: 11]
    )

    # Set shape information which is lost by py_function
# [source: 12]     # Shape: (num_sequences, sequence_length, height, width, channels)
    sequences.set_shape([None, sequence_length, img_height, img_width, 3]) # [cite: 12]

    # Repeat the label for each sequence extracted from the video
    num_sequences = tf.shape(sequences)[0]
    repeated_labels = tf.repeat(label, num_sequences) # [cite: 12]

    return sequences, repeated_labels

# --- Build Model (Keep the build_model function as it is, needed for loading) ---
def build_model(sequence_length, img_height, img_width, lstm_units, dense_units, dropout_rate,
                num_classes, learning_rate): # Add learning_rate back if needed for compilation during loading
# [source: 13]     """Builds the CNN + LSTM model."""
    input_shape = (sequence_length, img_height, img_width, 3)
    video_input = Input(shape=input_shape, name='video_input') # [cite: 13]

    # Load pre-trained CNN (without top classification layer)
    base_model = PRETRAINED_MODEL(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3) # [cite: 13]
    )
    base_model.trainable = False

    # --- Feature Extraction ---
# [source: 14]     # Create a model that outputs the features from the chosen layer
    intermediate_model = Model(inputs=base_model.input,
                               outputs=base_model.get_layer(FEATURE_EXTRACTOR_LAYER).output,
                               # Use the corrected layer name
                               name="intermediate_feature_extractor") # [cite: 15]

    # Wrap the intermediate model AND a pooling layer with TimeDistributed
    # This applies the feature extractor and pooling to each frame (time step)
    time_distributed_features = TimeDistributed(
        keras.Sequential([  # Add a Sequential wrapper
            intermediate_model, # [cite: 15]
            GlobalAveragePooling2D()  # Add pooling here to flatten spatial dimensions # [cite: 15]
        ]),
        name='time_distributed_features' # [cite: 16]
    )(video_input) # [cite: 16]

    # --- Temporal Learning (LSTM) ---
    # Output of TimeDistributed(GlobalAveragePooling2D) should be (batch, seq_len, features)
    # which is suitable for LSTM
    lstm_out = LSTM(lstm_units, return_sequences=False, name='lstm_layer')(time_distributed_features) # [cite: 16]
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # --- Classification Head ---
    x = Dense(dense_units, activation='relu', name='dense_1')(lstm_out)
    x = Dropout(dropout_rate)(x)
# [source: 17]     output = Dense(num_classes, activation='sigmoid', name='output_layer')(x) # num_classes should be 1 for binary

    # --- Compile Model ---
    model = Model(inputs=video_input, outputs=output, name='CrimePredictor') # [cite: 17]
    # Optimizer is needed for loading, but learning rate might not be critical if just predicting
    optimizer = Adam(learning_rate=learning_rate if learning_rate else 0.001) # Provide a default if needed
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', # [cite: 17]
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'), # [cite: 18]
                           tf.keras.metrics.Recall(name='recall')]) # [cite: 18]

    # print("Model Built Successfully:") # Comment out for Streamlit app
    # model.summary(expand_nested=True) # Comment out for Streamlit app
    return model


# --- Function to load the trained Keras model ---
@st.cache_resource
def load_keras_model(model_path="crime_predictor_best_generator.h5"): # Default path
    print(f"Loading model from: {model_path}")
    try:
        model = keras.models.load_model(model_path) # This line fails
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}") # You are seeing this error
        print(f"Error loading model {model_path}: {e}")
        return None

# How it's called in the Streamlit section
model = load_keras_model() # Uses the default path

# --- Function to process a single video file for prediction (Keep as is) ---
def load_single_video_for_prediction(video_path, sequence_length, img_height, img_width):
    """Loads frames from a single video path for prediction."""
    # This is similar to the _load_frames in load_and_process_video
    # but returns directly, not wrapped in tf.py_function
# [source: 35]     sequences = []
    cap = cv2.VideoCapture(video_path) # [cite: 35]
    if not cap.isOpened(): return [] # [cite: 35]
    frames = []
    try:
        while True:
            ret, frame = cap.read() # [cite: 35]
            if not ret: break
            resized_frame = cv2.resize(frame, (img_width, img_height)) # [cite: 36]
            normalized_frame = resized_frame / 255.0 # [cite: 36]
            frames.append(normalized_frame) # [cite: 36]
            if len(frames) == sequence_length:
                sequences.append(np.array(frames)) # [cite: 36]
# [source: 37]                 frames = frames[sequence_length // 2:] # Overlap [cite: 9, 37]
    finally:
        cap.release() # [cite: 10, 37]
    return np.array(sequences, dtype=np.float32) if sequences else None # [cite: 11, 37]


# --- Streamlit App ---
st.title("Crime Detection from Video Footage")
st.write("Upload a video file to predict if it contains crime-related activity.")

# Load the trained model
# Make sure 'crime_predictor_best_generator.keras' is accessible [cite: 25]
model = load_keras_model()

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

if uploaded_file is not None and model is not None:
    # Display the video
    st.video(uploaded_file)

    if st.button('Predict Activity'):
        with st.spinner('Processing video and making prediction...'):
            # Save temporary file to pass its path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                temp_video_path = tmpfile.name

            # Process the video
            sequences_np = load_single_video_for_prediction(
                temp_video_path, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH
            )

            if sequences_np is not None and len(sequences_np) > 0: # [cite: 37, 38]
                # Make prediction
                predictions = model.predict(sequences_np) # [cite: 37, 38]
                avg_prediction = np.mean(predictions) # [cite: 38]
                final_label = "Crime" if avg_prediction > 0.5 else "Normal" # [cite: 38]

                # Display result
                st.write(f"Average Predicted Probability: {avg_prediction:.4f}") # [cite: 39]
                if final_label == "Crime":
                    st.error(f"Prediction: {final_label}") # [cite: 38, 39]
                else:
                    st.success(f"Prediction: {final_label}") # [cite: 38, 39]
            else:
                st.warning(f"Could not extract valid sequences from the uploaded video.") # [cite: 39]

            # Clean up temporary file
            os.remove(temp_video_path)
elif model is None:
    st.error("Model could not be loaded. Please ensure the model file is available.")

# --- Original Main Execution Block (Commented out for Streamlit App) ---
# This part contains the training and evaluation logic from your original script.
# Keep it commented out or remove it if this script is *only* for the Streamlit app.

if __name__ == "__main__":
    [source: 19] print("TensorFlow Version:", tf.__version__)
    [source: 19] print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    1. Create list of all video paths and labels
    print("--- Gathering Video Files ---")
    [source: 19] all_video_data = list(video_generator(DATASET_PATH, CRIME_FOLDERS, NORMAL_FOLDER if os.path.exists(os.path.join(DATASET_PATH, NORMAL_FOLDER)) else None))
    if not all_video_data:
        print("Error: No video files found. Check dataset paths and permissions.") # [cite: 20]
        exit() # [cite: 20]

    [source: 20] all_video_paths = [item[0] for item in all_video_data]
    [source: 20] all_labels = [item[1] for item in all_video_data]

    print(f"Found {len(all_video_paths)} total videos.") # [cite: 20]
    print(f"Class Distribution: Normal (0): {all_labels.count(0)}, Crime (1): {all_labels.count(1)}") # [cite: 20]

    Split file paths and labels
    print("\n--- Splitting Data (File Paths) ---") # [cite: 20]
    try:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_video_paths, all_labels,
            test_size=0.25, random_state=42, stratify=all_labels # [cite: 20]
        )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels,
            test_size=0.20, random_state=42, stratify=train_labels # [cite: 21]
        )
    except ValueError as e: # [cite: 21]
        print(f"Error during splitting: {e}") # [cite: 21]
        print("Check if all classes have enough videos.") # [cite: 21]
        exit() # [cite: 21]

    [source: 21] print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}") # [cite: 22]
    print(f"Test set size: {len(test_paths)}") # [cite: 22]
    del all_video_data, all_video_paths, all_labels # Clear memory [cite: 22]


    3. Create tf.data Datasets
    print("\n--- Creating Data Generators ---") # [cite: 22]

    Training Dataset
    [source: 22] train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths)) # Shuffle paths before loading
    Use flat_map to handle videos producing multiple sequences or no sequences
    train_ds = train_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
        load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)) # [cite: 22]
    )
    train_ds = train_ds.batch(BATCH_SIZE) # [cite: 23]
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # Prefetch for performance [cite: 23]

    # Validation Dataset
     [source: 23] val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
     val_ds = val_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
         load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)) # [cite: 23]
     )
     val_ds = val_ds.batch(BATCH_SIZE)
     val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) # [cite: 24]

     Test Dataset
     [source: 24] test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
     test_ds = test_ds.flat_map(lambda path, label: tf.data.Dataset.from_tensor_slices(
         load_and_process_video(path, label, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)) # [cite: 24]
     )
     test_ds = test_ds.batch(BATCH_SIZE)
     test_ds = test_ds.prefetch(buffer_size=AUTOTUNE) # [cite: 24]


    # 4. Build Model
     print("\n--- Building Model ---") # [cite: 24]
     Assuming you want to build/train here too, otherwise load pre-trained
     model = build_model(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH,
                           LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE,
                           NUM_CLASSES, LEARNING_RATE) # [cite: 25, 28]

     5. Train Model using the datasets
     print("\n--- Starting Training ---") # [cite: 25]
     checkpoint_path = "crime_predictor_best_generator.keras" # New checkpoint name [cite: 25, 28]
     model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                        save_best_only=True, # [cite: 26, 29]
                                        monitor='val_accuracy', # [cite: 26, 29]
                                        mode='max', # [cite: 27, 30]
                                        verbose=1) # [cite: 27]
     early_stopping = EarlyStopping(monitor='val_loss', # [cite: 27, 30]
                                    patience=10, # [cite: 27, 30]
                                    mode='min', # [cite: 28, 31]
                                    restore_best_weights=True, # [cite: 28, 31]
                                    verbose=1) # [cite: 29, 32]

     history = model.fit(
         train_ds,
         epochs=EPOCHS,
         validation_data=val_ds,
         callbacks=[model_checkpoint, early_stopping],
         verbose=1 # [cite: 29]
     ) # [cite: 30, 32]

     print("\n--- Training Complete ---") # [cite: 30]
     try:
          pass # plot_training_history(history) # [cite: 18, 30, 32] # Implement or remove plotting
     except Exception as plot_err:
         print(f"Could not plot training history: {plot_err}") # [cite: 30]


    # 6. Evaluate Model
     print("\n--- Evaluating on Test Set using Generator ---") # [cite: 30]
     print(f"Loading best model from: {checkpoint_path}") # [cite: 30]
     try:
         best_model = keras.models.load_model(checkpoint_path) # [cite: 31, 33]
        # Evaluate using the test dataset generator
         loss, accuracy, precision, recall = best_model.evaluate(test_ds, verbose=1) # [cite: 31, 33]
         print(f"Test Loss: {loss:.4f}") # [cite: 31]
         print(f"Test Accuracy: {accuracy:.4f}") # [cite: 31]
         print(f"Test Precision: {precision:.4f}") # [cite: 31]
         print(f"Test Recall: {recall:.4f}") # [cite: 31, 34]

         # Detailed ClasFsification Report
         print("\nGenerating Classification Report (may take time)...") # [cite: 32]
         y_true = []
         y_pred_prob = []
         for batch_x, batch_y in test_ds: # [cite: 32]
              y_true.extend(batch_y.numpy()) # [cite: 32]
              batch_pred = best_model.predict(batch_x, verbose=0) # [cite: 32]
              y_pred_prob.extend(batch_pred.flatten()) # [cite: 32]

         y_pred = (np.array(y_pred_prob) > 0.5).astype(int) # [cite: 33]
         y_true = np.array(y_true) # [cite: 33]

         print("\nClassification Report:") # [cite: 33]
         print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Crime (1)'])) # [cite: 33, 34]

         print("\nConfusion Matrix:") # [cite: 33]
         print(confusion_matrix(y_true, y_pred)) # [cite: 33, 34]

     except Exception as e: # [cite: 33]
         print(f"Error during evaluation with best model: {e}") # [cite: 33, 35]
         Optionally, try evaluating with the model state at the end of training
         print("Evaluating with model state at end of training...") # [cite: 34]
         loss, accuracy, precision, recall = model.evaluate(test_ds, verbose=1) # [cite: 34, 36]
         print(f"Test Loss (End of Training): {loss:.4f}") # [cite: 34]
         print(f"Test Accuracy (End of Training): {accuracy:.4f}") # [cite: 34]
         print(f"Test Precision (End of Training): {precision:.4f}") # [cite: 34]
         print(f"Test Recall (End of Training): {recall:.4f}") # [cite: 34]

     7. Example Prediction (This logic is now integrated into the Streamlit part)
     print("\n--- Example Prediction ---") # [cite: 34]
     test_video_path = "/content/drive/MyDrive/RoadAccidents105_x264.mp4" # Keep your test path [cite: 37]
     if os.path.exists(test_video_path): # [cite: 37]
         test_sequences_np = load_single_video_for_prediction(test_video_path, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH) # [cite: 34, 37]
         if test_sequences_np is not None and len(test_sequences_np) > 0: # [cite: 37, 38]
             print(f"Predicting on video: {test_video_path} ({len(test_sequences_np)} sequences)") # [cite: 38]
             Use the loaded best_model (ensure it's loaded if running this block)
             best_model = load_keras_model() # Or however you load it here
             predictions = best_model.predict(test_sequences_np) # [cite: 37, 38]
             avg_prediction = np.mean(predictions) # [cite: 38]
             final_label = "Crime" if avg_prediction > 0.5 else "Normal" # [cite: 38]
             print(f"Average Predicted Probability: {avg_prediction:.4f}") # [cite: 39]
             print(f"Final Prediction for the video: {final_label}") # [cite: 38, 39]
         else:
             print(f"Could not extract valid sequences from {test_video_path}") # [cite: 39]
     else:
         print(f"Test video not found: {test_video_path}") # [cite: 39]

     print("\n--- Script Finished ---") # [cite: 39]
