# Visual Storytelling with Cross-Modal Learning (VIST)

This project implements a **Visual Storytelling system** using the **VIST (Visual Storytelling) dataset**, where a model generates a coherent story from a sequence of five images. The implementation focuses on a **memory-efficient CNN + LSTM baseline**, with clean dataset handling and a stable training pipeline in **Google Colab**.

---
## Quick Links
- [Experiments Notebook](experiment.ipynb) – Full experimental workflow and implementation  
- [Evaluation Results](results/) – All The results are in this folder 
- [Model Architecture](src/) – Encoders, fusion, temporal modelling, and decoders.

## 📌 Project Overview

Visual Storytelling goes beyond image captioning by requiring:
- temporal coherence across images  
- semantic consistency across sentences  
- long-form text generation  

This repository provides:
- a robust VIST dataset loader  
- a CNN + LSTM baseline model  
- memory-efficient training  
- optional cross-modal attention module (for future extensions)

---

## 📂 Dataset

**Dataset:** VIST – Stories in Sequence  
**JSON file used:** `train.story-in-sequence.json`

### Dataset Statistics
- 200,775 raw annotations  
- 167,528 images  
- 8,031 complete stories (5 images each)

### JSON Structure
```json
{
  "images": [...],
  "albums": [...],
  "annotations": [...]
}
````

### Dataset Processing

1. Load JSON file
2. Flatten nested annotations
3. Group by `album_id`
4. Sort by `worker_arranged_photo_order`
5. Keep only 5-image stories
6. Concatenate sentences
7. Pad / truncate to fixed length

---

## 🧾 Tokenization

Word-level tokenization is used.

Special tokens:

* `<pad>` = 0
* `<start>` = 1
* `<end>` = 2
* `<unk>` = 3

Vocabulary is built **only from the training set**.

---

## 🧠 Model Architecture

### 1️⃣ Image Encoder

* ResNet-50 (ImageNet pretrained)
* Fully frozen to save memory
* Output projected to embedding dimension

Input:

```
(B, 5, 3, 224, 224)
```

Output:

```
(B, 5, embed_dim)
```

Mean pooling is applied across the 5 images.

---

### 2️⃣ Text Encoder

* Word embedding layer
* Single-layer unidirectional LSTM

Output:

```
(B, L, hidden_dim)
```

---

### 3️⃣ Fusion (Baseline)

Late fusion using concatenation:

```
[text_features ; image_context] → Linear → ReLU → Dropout
```

This serves as the **baseline fusion strategy**.

---

### 4️⃣ Decoder

* Single-layer LSTM
* Step-by-step decoding (memory efficient)

Output:

```
(B, L, vocab_size)
```

---

## 🧪 Cross-Modal Attention (Optional)

A bi-directional cross-modal attention module is implemented:

* Text attends to images
* Images attend to text
* Padding-safe

> ⚠️ Not enabled in baseline training.
> Provided for future experimentation.

---

## ⚙️ Training Setup

### Objective

Next-token prediction with teacher forcing.

```
Input:  <start> w1 w2 ... w(n-1)
Target: w1 w2 ... w(n-1) <end>
```

### Loss

* CrossEntropyLoss
* Padding token ignored

### Optimizer

* Adam
* Learning rate: `3e-4`
* Weight decay: `1e-5`

### Scheduler

* Warm-up + decay using LambdaLR

### Regularization

* Dropout
* Gradient clipping

---

## 🖥️ Environment

* Platform: **Google Colab**
* GPU recommended (CUDA)
* CPU supported with smaller batch sizes
* DataLoader uses `num_workers = 0` for stability

---

## 🚀 Clone and Run (Google Colab)

### 1️⃣ Open Google Colab

Go to: [https://colab.research.google.com](https://colab.research.google.com)

---

### 2️⃣ Clone the Repository

Run in a Colab cell:

```bash
!git clone https://github.com/vishnureddy2349r-cell/CNN.git
%cd CNN
```

---

### 3️⃣ Install Dependencies

```bash
!pip install torch torchvision tqdm numpy pillow matplotlib
```

> PyTorch with CUDA is preinstalled in Colab.

---

### 4️⃣ Upload Dataset

Upload the dataset to Colab or Google Drive.

Recommended structure:

```
vist_kaggle/
├── train.story-in-sequence.json
├── images/
│   ├── <image_id>.jpg
│   └── ...
```

Mount Google Drive (optional):

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 5️⃣ Set Dataset Path

Update dataset path in code:

```python
dataset_path = "/content/vist_kaggle"
```

or

```python
dataset_path = "/content/drive/MyDrive/vist_kaggle"
```

---

### 6️⃣ Run Training

Run all notebook cells **from top to bottom**.

> Always restart runtime before re-training:

```
Runtime → Restart runtime
```

---

### 7️⃣ Training Output

* Training and validation loss per epoch
* Best model saved as:

```
best_model.pth
```

---

## 🧪 Sanity Check

```python
batch = next(iter(train_loader))
print(batch["images"].shape, batch["texts"].shape)
```

Expected:

```
(B, 5, 3, 224, 224)
(B, 200)
```

---

## Key Results
| Metric            | Score     |
|-------------------|-----------|
| Test Loss         | 7.8920    |
| Test Perplexity ↓ | 2675.9042 |


---

## 🔮 Future Work

* Sentence-level image alignment
* Beam search decoding

---

## 📚 Notes

* Dataset handling is correct and reproducible
* Training pipeline is stable in Colab
* Model is memory-efficient
* Suitable for coursework and research baselines

---

## 📄 License

For **educational and research use only**.
Follow the original VIST dataset license.

---
