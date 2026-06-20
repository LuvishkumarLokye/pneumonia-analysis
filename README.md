# Pneumonia Prediction from Chest X-Ray Images

A Flask and PyTorch web application that predicts whether a chest X-ray image is classified as `NORMAL` or `PNEUMONIA`. The project includes a trained convolutional neural network model, a browser-based upload interface, and a Jupyter notebook used for model development.

> Important: This project is for educational and research purposes only. It is not a medical device and should not be used as a substitute for professional clinical diagnosis.

## Features

- Chest X-ray image upload through a web dashboard
- Pneumonia vs normal classification
- Confidence score for the predicted class
- Separate probability scores for `NORMAL` and `PNEUMONIA`
- Custom CNN architecture built with PyTorch
- Image preprocessing with OpenCV and Torchvision transforms
- Flask API endpoint for inference
- Included notebook for model training and experimentation

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Flask |
| Deep Learning | PyTorch, Torchvision |
| Image Processing | OpenCV, NumPy |
| Frontend | HTML, CSS, JavaScript, jQuery |
| Model Development | Jupyter Notebook |

## Project Structure

```text
pneumonia_prediction/
|-- app.py
|-- best_pneumonia_model.pth
|-- pneumonia_model_creation.ipynb
|-- templates/
|   `-- dashboard.html
|-- Test sample images/
|-- Linkedin/
`-- README.md
```

## Model Overview

The application uses a custom convolutional neural network with:

- 4 convolutional layers
- Max pooling after each convolution block
- Dropout for regularization
- Fully connected classification layers
- 2 output classes: `NORMAL` and `PNEUMONIA`

Input images are resized to `224x224`, converted to tensors, and normalized before being passed to the model.

## Getting Started

### Prerequisites

- Python 3.8 or newer
- pip
- A trained model file named `best_pneumonia_model.pth`

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pneumonia_prediction.git
cd pneumonia_prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install flask torch torchvision opencv-python numpy
```

### Run the App

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:8080
```

Upload a chest X-ray image in PNG or JPEG format to generate a prediction.

## API Usage

The app exposes a prediction endpoint:

```text
POST /predict
```

Form field:

```text
image: chest X-ray image file
```

Example JSON response:

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 96.42,
  "prob_normal": 3.58,
  "prob_pneumonia": 96.42
}
```

## Dataset

The model notebook is designed around a chest X-ray pneumonia dataset with separate training, validation, and test folders.

Example dataset source:

```text
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Expected folder structure:

```text
chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- val/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

## Training

Open the notebook to review or retrain the model:

```bash
jupyter notebook pneumonia_model_creation.ipynb
```

After training, save the best model weights as:

```text
best_pneumonia_model.pth
```

The Flask app loads this file automatically on startup.

## GitHub Note

The included `best_pneumonia_model.pth` file is larger than GitHub's standard 100 MB file limit. If you publish this project to GitHub, use one of these options:

- Store the model with Git LFS
- Upload the model file as a GitHub Release asset
- Provide a download link and keep the model out of the repository

## Future Improvements

- Add `requirements.txt` for one-command dependency installation
- Add model evaluation metrics such as confusion matrix, precision, recall, and F1-score
- Add Grad-CAM heatmaps for visual explanation
- Add automated tests for the Flask prediction endpoint
- Add Docker support for easier deployment

## License

Add a license before publishing this repository if you want others to use or contribute to the project.
