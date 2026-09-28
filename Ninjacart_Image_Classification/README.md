# Ninjacart Fresh Produce Classifier

Multiclass image classification of **onion, potato, tomato** and **indian market (noise)** using CNNs and transfer learning (TensorFlow / Keras).

The notebook trains and compares five models, selects **MobileNetV2** as the final model, and evaluates it on a held-out test set.



## Business Problem

Ninjacart is India's largest fresh produce supply chain company. Sorting and grading vegetables by hand is slow and error-prone. This project builds an image classifier that can be plugged into an automated pipeline to identify the vegetable in an image, and to flag images that are not a target vegetable (the *indian market* / noise class).

## Dataset

Web-scraped images organised in `train/` and `test/` folders, one sub-folder per class. The notebook downloads and extracts the zip automatically from Google Drive.

| Class | Train | Test |
|---|---:|---:|
| indian market (noise) | 599 | 81 |
| onion | 849 | 83 |
| potato | 898 | 81 |
| tomato | 789 | 106 |
| **Total** | **3,135** | **351** |

- Classes are reasonably balanced (worst-case ratio about 1.5:1), so no class weighting was used.
- Images vary widely in size and aspect ratio (sampled: median about 300 x 202 px, min 48 px, max 2832 x 4256 px, aspect ratio 0.58 to 2.50). All images are resized to **224 x 224**.
- Note: the notebook's sanity check found the test set has 83 onion and 81 potato images, whereas the problem statement lists 81 and 83. Every other count matches.

## Data Split

- `train/` is split **80/20** into training (2,508 images) and validation (627 images) using `validation_split=0.2` with `seed=42`.
- `test/` (351 images) is kept completely held out for final evaluation.
- Batch size 32. Pipelines use `cache()` and `prefetch()`.

## Models

| # | Model | Notes |
|---|---|---|
| 1 | **Baseline CNN** | 3 x (Conv2D + MaxPool), Flatten, Dense(128). About 11.2M parameters. |
| 2 | **Improved CNN** | 4 conv blocks with BatchNorm and Dropout, GlobalAveragePooling, data augmentation. About 457K parameters. |
| 3 | **MobileNetV2** | ImageNet-pretrained, frozen base, custom head, then fine-tuned. **Selected model.** |
| 4 | **VGG16** | ImageNet-pretrained, frozen base, custom head. |
| 5 | **ResNet50** | ImageNet-pretrained, frozen base, custom head. |

**Data augmentation** (used by the Improved CNN and all transfer models, active only during training): random horizontal flip, rotation (0.15), zoom (0.15), contrast (0.15).

**Transfer-learning head:** `GlobalAveragePooling2D` -> `Dropout(0.4)` -> `Dense(128, relu)` -> `Dropout(0.3)` -> `Dense(4, softmax)`. Each backbone's own `preprocess_input` is built into the model.

**MobileNetV2 training:**
1. Head training with the base frozen: Adam, lr = 1e-3, up to 25 epochs (early stopping restored the epoch 9 weights).
2. Fine-tuning: top 30 layers of the base unfrozen, Adam lr = 1e-5, 10 epochs.

**Callbacks (all models):** EarlyStopping (`val_loss`, restore best weights), ModelCheckpoint, TensorBoard, ReduceLROnPlateau.

**Loss:** categorical cross-entropy. **Seed:** 42.

## Results

### Validation set (627 images)

| Model | Val Accuracy | Val Loss |
|---|---:|---:|
| **MobileNetV2** | **0.9713** | **0.0685** |
| ResNet50 | 0.9681 | 0.0838 |
| VGG16 | 0.9474 | 0.1369 |
| Improved CNN | 0.8915 | 0.2860 |
| Baseline CNN | 0.8325 | 0.4299 |

### Held-out test set (351 images)

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| ResNet50 | 0.9402 | 0.2247 |
| **MobileNetV2** | **0.9316** | **0.2270** |
| VGG16 | 0.9031 | 0.3115 |
| Improved CNN | 0.8348 | 0.4900 |
| Baseline CNN | 0.7949 | 0.4993 |

### Final model (MobileNetV2), per-class test performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| indian market | 1.00 | 0.88 | 0.93 | 81 |
| onion | 0.80 | 0.99 | 0.88 | 83 |
| potato | 0.96 | 0.84 | 0.89 | 81 |
| tomato | 1.00 | 1.00 | 1.00 | 106 |
| **Macro avg** | 0.94 | 0.93 | 0.93 | 351 |

## Why MobileNetV2?

MobileNetV2 was chosen because it combines:

- **The best validation accuracy and loss** of all five models (97.13% / 0.0685).
- **Deployment friendliness:** it is a lightweight architecture designed for mobile and edge inference, which suits Ninjacart's automated grading pipeline far better than the much heavier VGG16 or ResNet50.

On the test set, ResNet50 scored slightly higher (94.02% vs 93.16%), a gap of under one percentage point on 351 images. MobileNetV2 stays close while being the more practical model to deploy.

## Key Observations

- **Transfer learning clearly beats training from scratch.** All three pretrained models exceed 94% validation accuracy, versus 83% and 89% for the CNNs trained from scratch.
- **The baseline overfits.** Training accuracy reaches about 99% while validation accuracy plateaus near 83%, with rising validation loss.
- **Augmentation, BatchNorm and Dropout help** (83% -> 89% validation), but the Improved CNN's validation accuracy is unstable from epoch to epoch and its final result is well below the transfer models.
- **Tomato is easiest** (100% precision and recall on test). Errors come mostly from onion vs potato confusion: on test, onion has high recall but lower precision (0.80), and potato has lower recall (0.84), so some potatoes are being predicted as onion.
- **Validation-to-test gap:** MobileNetV2 drops from 97.1% (validation) to 93.2% (test), which suggests some distribution shift between the train and test images. Some *indian market* images are also being predicted as vegetables (recall 0.88).

## Tech Stack

Python, TensorFlow 2.20 / Keras, scikit-learn, NumPy, pandas, OpenCV, Matplotlib, Seaborn, TensorBoard, gdown. Developed in Google Colab on a GPU runtime.

## How to Run

1. Open `BusinessCase_Ninjacart_Vegetable_Classification.ipynb` in Google Colab (GPU runtime recommended).
2. Run all cells in order. The notebook installs `gdown`, downloads and extracts the dataset, trains all models, and produces the comparison tables and plots.
3. The final model is saved to `/content/MobileNetV2_final.keras`. Download it from the Colab file browser if you want to keep it.

To inspect training runs: `%load_ext tensorboard` then `%tensorboard --logdir /content/logs` (already included in the notebook).

## Using the Saved Model

The model expects raw RGB images of shape `224 x 224 x 3` with pixel values in `[0, 255]`. MobileNetV2 preprocessing is built in, so no extra normalisation is needed.

```python
import numpy as np
from tensorflow import keras

model = keras.models.load_model("MobileNetV2_final.keras")
class_names = ["indian market", "onion", "potato", "tomato"]  # alphabetical, as used in training

img = keras.utils.load_img("sample.jpg", target_size=(224, 224))
x = np.expand_dims(keras.utils.img_to_array(img), axis=0)

probs = model.predict(x)[0]
print(class_names[int(np.argmax(probs))], f"{probs.max():.2%}")
```

## Limitations and Future Work

- The data is web-scraped. Collecting real warehouse and market images would reduce the domain gap seen in the validation-to-test drop.
- The noise class covers only *indian market* scenes. Adding more diverse non-vegetable negatives would make the classifier more robust to arbitrary inputs in production.
- Onion vs potato is the main confusion; more data of those two classes (or higher input resolution) may help.
- Ensembling the top two models (MobileNetV2 and ResNet50) could give a small additional accuracy gain, at the cost of heavier inference.
- The test set was used to compare all models for reporting. Model selection itself was made on the validation set only.
