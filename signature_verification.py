# Step 1: Install necessary libraries
!pip install opencv-python-headless scikit-learn numpy

# Step 2: Import required libraries
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from google.colab import files
import matplotlib.pyplot as plt

# Step 3: Define functions for preprocessing and verification
def preprocess_image(image_path):
    """
    Preprocess the input image by converting to grayscale, resizing, and applying thresholding.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    # Resize to a standard size
    image = cv2.resize(image, (200, 100))
    # Apply binary thresholding
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_features(image):
    """
    Flatten the processed image into a feature vector.
    """
    return image.flatten()

def verify_signature(reference_path, test_path, threshold=0.1):
    """
    Compare the test signature with the reference signature and calculate similarity.
    """
    # Preprocess images
    ref_image = preprocess_image(reference_path)
    test_image = preprocess_image(test_path)

    # Extract features
    ref_features = extract_features(ref_image)
    test_features = extract_features(test_image)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(ref_features, test_features)

    # Convert MSE to similarity score
    similarity_score = 1 / (1 + mse)

    # Determine if signatures match
    is_match = similarity_score > threshold
    return is_match, similarity_score, ref_image, test_image

# Step 4: Upload images for verification
print("Upload the reference signature image:")
reference_file = files.upload()  # Upload reference_signature.jpg
reference_path = list(reference_file.keys())[0]

print("Upload the test signature image:")
test_file = files.upload()  # Upload test_signature.jpg
test_path = list(test_file.keys())[0]

# Step 5: Perform signature verification
try:
    threshold = 0.1  # Set similarity threshold
    match, score, ref_image, test_image = verify_signature(reference_path, test_path, threshold)

    # Step 6: Display results
    print(f"Similarity Score: {score:.2f}")
    if match:
        print("Result: Signatures Matched!")
    else:
        print("Result: Signatures Mismatched!")

    # Show reference and test images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Reference Signature")
    plt.imshow(ref_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Test Signature")
    plt.imshow(test_image, cmap="gray")
    plt.axis("off")

    plt.show()

except ValueError as e:
    print(f"Error: {e}")
