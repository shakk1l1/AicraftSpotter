import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import regex as re

from database import get_image_data


def load_images(image_dir, label_file):
    images = []
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            label = parts[1].removesuffix('\n')
            img_path = os.path.join(image_dir, parts[0] + '.jpg')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            values = get_image_data(parts[0])
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
            croped_img = cv2.resize(croped_img, (128, 128))  # Resize images to 128x128
            images.append(croped_img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

# Preprocess images
def preprocess_images(images):
    images = images / 255.0  # Normalize images
    mean_image = np.mean(images, axis=0)
    centered_images = images - mean_image
    return centered_images, mean_image

# Perform POD (PCA)
def perform_pod(images, n_components=50):
    pca = PCA(n_components=n_components)
    pca.fit(images)
    return pca

# Train classifier
def train_classifier(pca, images, labels):
    projected_images = pca.transform(images)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(projected_images, labels)
    return clf

# Evaluate classifier
def evaluate_classifier(clf, pca, images, labels):
    projected_images = pca.transform(images)
    predictions = clf.predict(projected_images)
    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Main function

train_images, train_labels = load_images('data/images', 'data/images_manufacturer_trainval.txt')
test_images, test_labels = load_images('data/images', 'data/images_manufacturer_test.txt')

centered_train_images, mean_image = preprocess_images(train_images)
centered_test_images = (test_images / 255.0) - mean_image

pca = perform_pod(centered_train_images)
clf = train_classifier(pca, centered_train_images, train_labels)

accuracy = evaluate_classifier(clf, pca, centered_test_images, test_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')