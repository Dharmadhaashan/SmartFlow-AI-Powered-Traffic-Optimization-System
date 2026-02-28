import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DATASET_PATH = r'D:\project-traffic' # Path to your unzipped Kaggle folder
CLASSES = ['ambulance', 'firetruck', 'police', 'traffic']
IMG_HEIGHT = 40  # Number of MFCCs (Frequencies)
IMG_WIDTH = 130  # Time steps (Approx 3 seconds)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=IMG_HEIGHT)
        # Ensure consistent width
        if mfccs.shape[1] < IMG_WIDTH:
            mfccs = np.pad(mfccs, ((0,0), (0, IMG_WIDTH - mfccs.shape[1])))
        else:
            mfccs = mfccs[:, :IMG_WIDTH]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 2. DATA LOADING & AUTOMATIC SPLIT ---
X, y = [], []
print("🔍 Loading data and extracting features...")

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder):
        print(f" Warning: Folder {folder} not found. Skipping.")
        continue
    
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            feat = extract_features(os.path.join(folder, file))
            if feat is not None:
                X.append(feat)
                y.append(label)

X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y = np.array(y)

# Encode class names to numbers
le = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(le.fit_transform(y))

# Split data: 80% for training, 20% for testing accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- 3. MODEL ARCHITECTURE ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAINING ---
print("\n Starting Training...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# --- 5. EVALUATING TESTING ACCURACY ---
print("\n Evaluating on Test Set...")
scores = model.evaluate(X_test, y_test, verbose=0)
print(f" Final Testing Accuracy: {scores[1]*100:.2f}%")

# Generate Detailed Report & Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

# Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix: Predicted vs Actual')
plt.show()

# Save the model for your real-time script
model.save('siren_detector.h5')
np.save('classes.npy', le.classes_)
print("\n Model saved as 'siren_detector.h5'")