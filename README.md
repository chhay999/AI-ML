# CNN Image Classification: Animal Predictor

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify animal images across 5 categories:
- 🐱 **Cat**
- 🐮 **Cow**
- 🐶 **Dog**
- 🐘 **Elephant**
- 🐴 **Horse**

---

## 📁 Project Structure

```
CNN-on-the-prediction-of-image/
│
├── Dataset/                      # All images standardized to (200, 200, 3) RGB
│   ├── train/                    # 3,000 training images (600 per class)
│   ├── validation/               # 500 validation images (100 per class)
│   └── Test/                     # 500 test images (100 per class)
│
├── models/
│   ├── CNN_predict_model.keras   # Final trained model (native Keras format)
│   ├── CNN_predict_model.h5      # Final trained model (HDF5 format)
│   └── CNN_best_model.keras      # Best checkpoint based on validation accuracy
│
├── resize_dataset.py             # Preprocessing script to resize images to 200x200 RGB
├── train.py                      # Training script with multi-stage Dropout, BatchNorm, and callbacks
├── predict.py                    # Inference script for predicting single images or directories
├── CNN.ipynb                     # Interactive Jupyter Notebook
├── gpu_config.py                 # GPU memory growth & CUDA configuration
├── history.csv                   # Epoch-by-epoch loss & accuracy log
└── training_accuracy_loss.png    # Training vs Validation curves
```

---

## 🚀 How to Run the Project

### 1. Activate Virtual Environment
Activate the preconfigured virtual environment:
```bash
source venv/bin/activate
```
*(Or invoke scripts directly using `./venv/bin/python <script_name>.py`)*

---

### 2. (Optional) Resize Dataset Images to 200×200
All images currently in `Dataset/` are already resized to 200×200 RGB. If you ever add new images, run:
```bash
python resize_dataset.py
```

---

### 3. Train the Model

#### A. Regularized Deep CNN (Multi-Stage Dropout + Batch Normalization)
```bash
python train.py --epochs 25 --batch_size 32 --dropout 0.4
```

#### B. Transfer Learning with MobileNetV2 (High Accuracy: >90%)
```bash
python train.py --epochs 15 --batch_size 32 --dropout 0.4 --model_type mobilenet
```

**Available Training Options:**
- `--epochs`: Number of epochs (default: `25`)
- `--batch_size`: Batch size (default: `32`)
- `--dropout`: Dropout probability for dense head and intermediate blocks (default: `0.4`)
- `--model_type`: Choose `'cnn'` (default) or `'mobilenet'` (transfer learning)
- `--learning_rate`: Initial learning rate (default: `0.001`)
- `--no_augment`: Flag to disable data augmentation (augmentation is ON by default)

---

### 4. Run Predictions & Test the Model

#### Predict on a sample image from each class automatically:
```bash
python predict.py
```

#### Predict on a single image file:
```bash
python predict.py --image Dataset/Test/cat/0.jpg
```

#### Predict on an entire folder of images:
```bash
python predict.py --dir Dataset/Test/elephant
```

---

### 5. Run with Jupyter Notebook
Open [`CNN.ipynb`](CNN.ipynb) in VS Code or Jupyter Lab, select the Python environment at `./venv/bin/python`, and run the cells sequentially.
# AI-ML
