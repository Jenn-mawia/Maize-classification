# A. DATA IMPORTS

# data manipulation
import pandas as pd
import numpy as np 

# visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# modeling

import os
import cv2


import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



# DATA LOADING

# Set the base directory of your dataset
base_dir = "/Users/jeniphermawia/Desktop/crop"

data = []
for folder in os.listdir(base_dir):
    if folder.startswith("Maize"):
        label = folder.replace("Maize___", "")
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = os.path.join(folder_path, file)
                data.append([image_path, label])

# Create a dataframe
df = pd.DataFrame(data, columns=["image_path", "label"])

# Save to CSV
df.to_csv("maize_dataset.csv", index=False)

# DATA PREPROCESSING

# remove spaces on the file paths
df['image_path'] = df['image_path'].str.strip()


# get the data in colored format
colored_images = []
y_colored_labels = []
for _, row in df.iterrows():
    path = row['image_path']
    label = row['label']
    try:
        img = Image.open(path).convert('RGB')  # COLORED images
        img = img.resize((64, 64)) # already tried (128,128), (28,28)
        colored_images.append(np.array(img))
        y_colored_labels.append(label)
    except:
        continue

# Convert to numpy arrays
X_colored = np.array(colored_images).reshape(-1, 64, 64, 3) / 255.0  # Normalize

# Encode class labels to integers
le = LabelEncoder()
y_colored_integer = le.fit_transform(y_colored_labels)
y_colored_categorical = to_categorical(y_colored_integer)

# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X_colored, y_colored_categorical, test_size=0.2, random_state=42, stratify=y_colored_categorical)

# TRAINING MODEL


model2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
history2 = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32) # train for longer, more epochs


# MAKING PREDICTIONS
y_pred2 = model2.predict(X_test)
y_pred_labels2 = np.argmax(y_pred2, axis=1)
y_true_labels2 = np.argmax(y_test, axis=1)
# Kindly add these two lines at the end of maize.py to save the model and label encoder for API predictions
model2.save("maize_model2.h5")
import joblib
joblib.dump(le, "label_encoder.pkl")
