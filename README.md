# 🍠 Potato Disease Classification using Deep Learning

This project focuses on building a Convolutional Neural Network (CNN) model to classify potato plant leaves as **Healthy**, **Early Blight**, or **Late Blight**. It uses TensorFlow and Keras to train the model on an image dataset and evaluates its performance using various metrics.

---

## 📂 Project Structure

- `potato-disease-classification-model.ipynb`: Jupyter Notebook containing data preprocessing, model training, evaluation, and visualization steps.
- `README.md`: Project overview and setup instructions.
- `dataset/`: (User must provide) Folder containing the `train`, `test`, and `validation` image directories organized by class.

---

## 📊 Dataset

The dataset used is from the **PlantVillage** dataset (hosted on Kaggle), structured in the following format:

Potato/

├── train/

│ ├── Early_Blight/

│ ├── Late_Blight/

│ └── Healthy/

├── test/

└── val/


- Classes: `Early_Blight`, `Late_Blight`, `Healthy`
- Format: JPEG/PNG images of potato leaves

Download: [PlantVillage Dataset - Potato](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) *(or user-defined source)*

---

## 🚀 Features

- Image preprocessing using ImageDataGenerator
- CNN model built with TensorFlow and Keras
- Accuracy and loss visualization
- Confusion matrix and classification report
- Model saving using `.h5` format

---

## 🛠️ Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/CodeHive08/potato-disease-classification.git
   cd potato-disease-classification
## Install required packages:
install the key libraries:

pip install tensorflow matplotlib seaborn scikit-learn
Download and set up the dataset as per the folder structure mentioned above.

## 🧪 Model Training & Evaluation
Run the notebook using Jupyter:

jupyter notebook potato-disease-classification-model.ipynb
Or use Google Colab by uploading the notebook and mounting your dataset folder.

## 📈 Results
Training Accuracy: ~95–99%

Validation Accuracy: ~92–96%

Final Model Saved as: potato_disease_model.h5

Confusion matrix and classification report are plotted and printed inside the notebook for detailed analysis.



## 🔮 Future Enhancements
Convert to a web app using Flask/Streamlit for real-time prediction

Use data augmentation for better generalization

Deploy model using TensorFlow Lite or ONNX for mobile devices

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the MIT License.

## 🙌 Acknowledgements
-PlantVillage Dataset on Kaggle

-TensorFlow and Keras documentation

GitHub Copilot and ChatGPT for productivity assistance
