# CIFAR10-Image-Classifier
A CIFAR-10 image classification project using a custom CNN with data augmentation. Includes training scripts, preprocessing tools, a saved model, and a Streamlit web app for real-time predictions. Supports testing with generated CIFAR-10 images.
---
##  Project Overview  
This project uses a custom-built Convolutional Neural Network trained on the **CIFAR-10** dataset.  
It includes:

- A **CNN training pipeline** with data augmentation  
- A **Streamlit-based web app** for real-time image classification  
- **Utility functions** for preprocessing  
- **Test images generator** for evaluation  
- A saved trained model (`cifar10_model.h5`)  
- A performance comparison table of multiple ML models

The classifier predicts one of the following 10 classes:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`
---
## Model Accuracy  
Based on the accuracy results file:

| Model                | Accuracy (%) | Notes                              |
| -------------------- | ------------ | ---------------------------------- |
| Logistic Regression  | 45.3         | Baseline, fast but low accuracy    |
| Naive Bayes          | 43.1         | Poor with complex images           |
| SVM                  | 58.4         | Better but slow training           |
| Decision Tree        | 53.7         | Overfits easily                    |
| Random Forest        | 61.9         | Improved generalization            |
| Gradient Boosting    | 64.8         | Good balance of speed and accuracy |
| KNN                  | 68.5         | Works well with small data         |
| CNN (This model)     | 83.7         | High performance on image data     |
| **Ensemble (CNN + KNN)** | **85.9** | Best accuracy, reduced confusion   |

*Source: `accuracy.txt`*
---
## 🚀 Features  
✔ Custom CNN architecture with batch normalization & dropout  
✔ Data augmentation for improved generalization  
✔ Streamlit interface for easy testing  
✔ Confidence score and probability chart  
✔ Generates test images automatically  
✔ Clean and modular code structure  

---
