# Support Vector Machine (SVM) - Cat vs Dog Image Classification

import os
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# 1. Set dataset path
# ---------------------------------------------------

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Folder structure:
#
# dataset/
# ├── cats/
# │   ├── cat1.jpg
# │   ├── cat2.jpg
# │   └── ...
# │
# └── dogs/
#     ├── dog1.jpg
#     ├── dog2.jpg
#     └── ...

# ---------------------------------------------------
# 2. Load and preprocess images
# ---------------------------------------------------

IMG_SIZE = 64

images = []
labels = []

categories = ["cats", "dogs"]

missing_folders = [
    category
    for category in categories
    if not os.path.isdir(os.path.join(dataset_path, category))
]

if missing_folders:
    raise FileNotFoundError(
        "Dataset folders are missing: "
        + ", ".join(missing_folders)
        + f". Create {os.path.join(dataset_path, 'cats')} and "
        + f"{os.path.join(dataset_path, 'dogs')}, then add the images."
    )

for label, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        if not os.path.isfile(image_path):
            continue

        try:
            # Pillow supports common formats, including AVIF.
            image = Image.open(image_path).convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image = np.asarray(image, dtype=np.float32)

            # Normalize pixel values
            image = image / 255.0

            # Flatten image into one-dimensional array
            image = image.flatten()

            images.append(image)
            labels.append(label)

        except (OSError, ValueError) as error:
            print(f"Unable to read {image_name}: {error}")

# Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

if len(X) == 0 or len(np.unique(y)) < 2:
    raise ValueError("The dataset must contain readable images in both cats and dogs folders.")

class_counts = np.bincount(y)
if np.min(class_counts) < 2:
    raise ValueError("Each class must contain at least two readable images.")

print("Total images:", len(X))
print("Image feature size:", X.shape)

# ---------------------------------------------------
# 3. Split dataset into training and testing data
# ---------------------------------------------------

# Keep at least one test sample for each class when the dataset is small.
test_size = max(0.20, 2 / len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# 4. Scale the features
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# 5. Create SVM model
# ---------------------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

# ---------------------------------------------------
# 6. Train the model
# ---------------------------------------------------

model.fit(X_train, y_train)

# ---------------------------------------------------
# 7. Make predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# 8. Evaluate the model
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)

# ---------------------------------------------------
# 9. Predict a new image
# ---------------------------------------------------

image_path = input("\nEnter image path: ")

try:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = scaler.transform([image.flatten()])
    prediction = model.predict(image)
    print("Prediction:", "CAT" if prediction[0] == 0 else "DOG")
except (OSError, ValueError) as error:
    print("Image could not be loaded.")
    print(error)