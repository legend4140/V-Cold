import pandas as pd
import numpy as np
import pyaudio
import wave
import librosa
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def extract_features(audio_data, sr=44100, n_mfcc=None, max_features=93):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spectrogram_features = mel_spectrogram.T

    if n_mfcc is not None:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        mfcc_features = mfccs.T

        # Concatenate MFCC features and Mel-spectrogram features
        combined_features = np.concatenate([mfcc_features, mel_spectrogram_features], axis=1)

        # Check if the total features exceed the maximum allowed
        if combined_features.shape[1] > max_features:
            combined_features = combined_features[:, :max_features]

        return combined_features.reshape(1, -1)[:,:max_features]

    # Return only Mel-spectrogram features if n_mfcc is None
    return mel_spectrogram_features.reshape(1, -1)[:,:max_features]

# Load the preprocessed data
data = pd.read_csv('data_hsin.csv')

# Split the data into training and testing sets
X = data.drop('sick', axis=1).values
y = data['sick'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for Conv1D
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Use LabelEncoder to convert categorical labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to categorical format
num_classes = np.max(y_train_encoded) + 1
y_train_categorical = to_categorical(y_train_encoded, num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes)

# Define the model architecture
# Define the model architecture
enhanced_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes * 2, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])


# Compile the model
enhanced_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
enhanced_model.fit(X_train_reshaped, y_train_categorical, epochs=20, batch_size=64,
                    validation_data=(X_test_reshaped, y_test_categorical))

# Save the trained model as "cold_det.h5"
enhanced_model.save("cold_det.h5")

# Load the saved model
loaded_model = load_model("cold_det.h5")

# Recording audio from the user
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# ... (Previous code remains unchanged) ...

print("Finished recording...")
#print(X_train.shape[1])
# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded audio to a WAV file
wave_output_filename = "recorded_audio.wav"
with wave.open(wave_output_filename, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Preprocess the recorded audio data and extract features
audio_data, _ = librosa.load(wave_output_filename, sr=RATE)
audio_features = extract_features(audio_data, sr=RATE, n_mfcc=None)

# Check the number of features the model was trained on
num_trained_features = X_train.shape[1]
print(f"Number of features the model was trained on: {num_trained_features}")

# Check the number of features in audio_features
num_audio_features = audio_features.shape[1]
print(f"Number of features extracted from audio: {num_audio_features}")

# Ensure the audio features are in the expected shape
if audio_features.ndim == 2:  # 2D features (e.g., MFCCs)
    if num_audio_features != num_trained_features:
        raise ValueError(f"Number of features extracted ({num_audio_features}) "
                         f"does not match the expected number of features ({num_trained_features})")

    # Perform necessary transformations or scaling
    audio_features_scaled = scaler.transform(audio_features)
    audio_features_reshaped = audio_features_scaled.reshape(
        audio_features_scaled.shape[0], audio_features_scaled.shape[1], 1
    )
elif audio_features.ndim == 1:  # 1D features (e.g., Mel-spectrogram)
    # Perform necessary transformations or scaling
    # ...
    pass
else:
    raise ValueError(f"Unexpected audio features shape: {audio_features.shape}")

# Perform prediction on the preprocessed audio features using the loaded model
predicted_probs = loaded_model.predict(audio_features_reshaped)
predicted_label = np.argmax(predicted_probs, axis=1)

# Display the prediction result
print("Predicted Label:", predicted_label)

if np.any(predicted_label in [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92,
    93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 108, 109, 110, 111, 112, 113, 115, 118, 121, 122, 123,
    124, 125, 126, 127, 133, 144, 145, 146, 147, 152, 158, 159, 161, 162, 163, 164, 165, 166, 167, 170, 171, 174,
    175, 176, 177, 178, 180, 182, 185, 187, 188, 189, 190, 192, 194, 197, 199, 200, 201, 204, 207, 209, 210, 212,
    213, 216, 218, 219, 220, 224, 225, 230, 231, 232, 233, 234, 236, 239, 240, 241, 242, 243, 244, 245, 246, 249,
    250, 253, 255, 258, 260, 261, 263, 264, 265, 266, 267, 268, 271, 274, 275, 276, 277, 280, 281, 282, 284, 285,
    286, 287, 288, 289, 291, 293, 296, 297, 301, 302, 303, 306, 307, 308, 310, 311, 313, 314, 315, 316, 317, 318,
    319, 320, 321, 322, 323, 325, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343,
    344, 346, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 364, 365, 366, 367, 368, 369, 370,
    379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    414, 415, 416, 417, 418, 419, 420, 421, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
    448, 449, 450, 451, 452, 457, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 477, 485, 489,
    491, 492, 494, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 510, 514, 515, 516, 517, 518, 519, 525,
    527, 532, 538, 540, 542]):
    print("Healthy detected")
elif np.any(predicted_label in [
    140, 186, 193, 211, 238, 257, 298, 326, 390, 391, 411, 424, 425, 430,
    431, 432, 433, 453, 473, 474, 484, 511, 522, 529, 533, 536, 548]):
    print("Cough detected")
elif np.any(predicted_label in [
    119, 128, 129, 131, 132, 134, 137, 142, 168, 173, 181, 191, 195, 196,
    203, 221, 228, 229, 235, 237, 248, 252, 273, 283, 290, 472, 486]):
    print("Headache detected")
elif np.any(predicted_label in [
    45, 72, 99, 106, 107, 114, 116, 120, 150, 151, 154, 155, 160, 183,
    198, 214, 222, 259, 262, 279, 295, 299, 305, 309, 312, 345, 355, 371,
    393, 394, 412, 413, 427, 478, 482, 483, 488, 509, 513, 531, 537, 544,
    547]):
    print("Running nose detected")
elif np.any(predicted_label in [
    117, 149, 153, 157, 169, 179, 215, 223, 254, 256, 269, 292, 294, 304,
    324, 347, 378, 395, 408, 409, 410, 426, 428, 429, 454, 455, 456, 458,
    476, 480, 481, 487, 523, 524, 528, 534]):
    print("Sore Throat detected")
elif np.any(predicted_label in [
     41,  54,  58,  91, 130, 135, 138, 148, 208, 300, 333, 362, 363, 372,
     373, 374, 375, 376, 377, 392, 396, 422, 423, 475, 479, 507, 508, 512,
     526, 530, 541]):
     print("Sputum detected")
elif np.any(predicted_label in [
     136, 139, 141, 143, 156, 172, 184, 202, 205, 206, 217, 226, 227, 247,
     251, 270, 272, 278, 490, 493, 495, 520, 521, 535, 539, 543, 545, 546]):
     print("Stuffy nose detected")
else:
    print("Cannot process your data. Consider Re-recording!")
