# PR-Classification-with-MedSynGAN-Enhancing-Model-Performance-via-Synthetic-Histopathology

🌐 Project Overview
This repository presents a novel approach to PR (Progesterone Receptor) classification in immunohistochemistry (IHC) images using a two-part pipeline:

1. A Simple CNN for binary classification of PR+ and PR− images.
2. A Conditional GAN (MedSynGAN) designed to generate synthetic pathology samples to aid in data augmentation and model robustness.
Both components are implemented in a reproducible and explainable Jupyter notebook executed in a GPU-enabled Kaggle environment.

🗂 Dataset Summary (Private Data, will be share upon request)
| Class  | Description                          | Source Path                         |
| ------ | ------------------------------------ | ----------------------------------- |
| PR+    | Positive IHC stained slides          | `/kaggle/input/pr-positive-ihc/PR+` |
| PR−    | Negative IHC stained slides          | `/kaggle/input/pr-negative-ihc/PR-` |
| Format | RGB `.jpg` / `.png` histology images | Size: `128x128` resized             |

All images were shuffled and split with a 70/30 ratio into train/ and test/ folders using stratified sampling.

🧠 # Model 1: Simple CNN for Binary Classification
A lightweight convolutional neural network was implemented to serve as a fast and effective baseline classifier.

Architecture Summary
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Flatten → Linear → ReLU → Linear(2)

Training Metrics
| Metric        | Value (Epoch 50) |
| ------------- | ---------------- |
| Accuracy      | \~93.5%          |
| Loss          | Reduced steadily |
| Optimizer     | Adam             |
| Loss Function | CrossEntropy     |

📊 Performance Evaluation
🔹 Training Curves
<p align="center"> <img src="figures/training_curves.png" alt="Training Loss and Accuracy" width="600"/> </p>

🔹 Confusion Matrix
|            | Predicted PR− | Predicted PR+ |
| ---------- | ------------- | ------------- |
| Actual PR− | 35            | 2             |
| Actual PR+ | 3             | 40            |


🎨 Model 2: MedSynGAN for Data Synthesis
To mitigate dataset imbalance and increase generalizability, a Conditional GAN (cGAN) named MedSynGAN was introduced.

Generator Design
-Noise vector + label embedding → FC → Reshape → 3 Transpose Convs → 128x128 RGB output
-Uses ReLU activations and Tanh for output normalization.

Discriminator Design
-Input image + label embedding as an additional channel
-Series of Conv2D layers + LeakyReLU + Sigmoid output

| Component     | Description                     |
| ------------- | ------------------------------- |
| Input Size    | 100-dim noise + label embedding |
| Output Image  | 3x128x128                       |
| Optimizer     | Adam (`lr=0.0002`)              |
| Loss Function | Binary Cross-Entropy            |

💡 Key Takeaways
-The SimpleCNN achieves strong baseline classification accuracy for PR-stained IHC slides.
-MedSynGAN enables effective synthetic image generation, which is promising for data augmentation in future experiments.
-Visualizations such as confusion matrices and ROC curves provide transparency into the model’s behavior.

🔁 Future Work
-Integrate synthetic samples during CNN training for performance benchmarking.
-Expand to multi-class hormone receptor classification (e.g., ER, HER2).
-Apply advanced interpretable AI (e.g., Grad-CAM, LIME) for model transparency.

📁 Directory Structure
📦PR-MedSynGAN
 ┣ 📄 pr-classification-medsyngan.ipynb
 ┣ 📁 figures/
 ┃ ┣ 📄 training_curves.png
 ┃ ┣ 📄 confusion_matrix.png
 ┃ ┗ 📄 roc_curve.png
 ┗ 📄 README.md





