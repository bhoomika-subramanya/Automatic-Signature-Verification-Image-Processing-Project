# ✍️ Automatic Signature Verification using Digital Image Processing

## 🧾 Project Overview

Automatic Signature Verification is an essential tool for validating handwritten signatures in legal, banking, and secure documentation systems. This project implements a simple yet effective image-processing-based signature verification system using Python, OpenCV, and Scikit-learn.

The system compares a reference signature against a test signature using **preprocessing**, **feature extraction**, and **similarity scoring** (via Mean Squared Error). It then predicts whether the two signatures match.

---

## 🎯 Features

- Signature comparison using image preprocessing
- Automatic threshold-based similarity measurement
- Mean Squared Error-based feature distance scoring
- Simple, interpretable output (match / mismatch)
- Visual display of compared signatures using Matplotlib

---

## 🛠️ Tech Stack

| Component         | Technology             |
|------------------|------------------------|
| Programming Lang | Python 3.x             |
| Libraries Used   | OpenCV, Scikit-learn, NumPy, Matplotlib |
| Platform          | Google Colab / Jupyter Notebook |

---

## 📁 Files and Structure

📂 signature-verification
├── signature_verification.ipynb # Main notebook with implementation
├── reference_signature.jpg # Sample reference signature
├── test_signature.jpg # Sample test signature
└── README.md # Project documentation

🚀 How It Works
1. Preprocessing
Convert image to grayscale
Resize to a fixed shape (200x100)
Apply Otsu's binarization to highlight the signature pattern

2. Feature Extraction
Flatten the binarized image into a 1D vector for comparison

3. Similarity Measurement
Compute Mean Squared Error (MSE) between vectors
Convert MSE into similarity score:
similarity = 1 / (1 + MSE)
Classify as Matched if similarity > threshold (e.g., 0.1)

🔧 Usage Instructions
Upload the reference signature and test signature images via files.upload() in Colab or use your own input method.

Run all cells to perform verification.

View results in the console and side-by-side signature images for comparison.

📷 Example Output
Upload the reference signature image:
Upload the test signature image:
Similarity Score: 0.89
Result: Signatures Matched!
