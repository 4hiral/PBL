# Age and Gender Detection using VGGFace: A Deep Learning-Based Facial Attribute Estimation

This project applies deep learning techniques to estimate **age** and classify **gender** from facial images using the **VGGFace model** with **transfer learning**. Designed for applications in **smart systems**, **surveillance**, **personalization**, and **demographic analytics**, the model is optimized to handle variations in lighting, pose, and facial features for reliable predictions.


🚀 **Project Overview**

Accurate age and gender estimation is crucial for modern AI systems in areas such as security, targeted advertising, and human-computer interaction. This project aims to:

- Utilize the pre-trained VGGFace CNN for facial feature extraction.
- Fine-tune the model for simultaneous age regression and gender classification.
- Apply data preprocessing and augmentation for better generalization.
- Use the IMDB-WIKI dataset for training and UTKFace for cross-validation.
- Visualize predictions, evaluate performance metrics, and analyze errors.


🔧 **Technologies Used**
- Python
- TensorFlow / Keras
- OpenCV (for image preprocessing)
- NumPy, Pandas (for data handling)
- Matplotlib, Seaborn (for visualization)
- Jupyter Notebook (for development and experimentation)
  
📂 **Project Structure**
├── model/
│ └── final_age_gender_model.h5 # Trained Keras model
├── data/
│ └── UTKFace/ # Evaluation dataset
│ └── IMDB-WIKI/ # Training dataset
├── notebooks/
│ └── VGGFace_FineTuning.ipynb # Jupyter notebook for model training and evaluation
├── report/
│ └── Fine_Tune_CNN_Age_Gender.pdf # Research paper/documentation
├── README.md
└── requirements.txt # Python dependencies
