## 🖐️ Hand Pose Estimation with PyTorch

This project focuses on predicting **21 hand keypoints (x, y)** from RGB images using a **Convolutional Neural Network (CNN)** trained on FreiHAND_pub_v2 dataset. It lays the foundation for real-time gesture-based control systems (an Upcoming Project - **HOLOCONTROL**) .

---

### Dataset Description
  Source: FreiHAND dataset
  Type: RGB images with annotated 2D keypoints
  Usage: We use the public subset containing ~32,000 images with corresponding joint labels.

* **Images:**

  * Total: **32,560**
  * Size: Resized to **128x128**
  * Format: `.jpg`

* **Keypoints:**

  * 21 keypoints per hand, flattened to 42 values (x1, y1, x2, y2, ..., x21, y21)
  * Normalized in the range `[0, 1]` using image width and height

---

### Model Architecture

Built using PyTorch:

```text
Input: (3, 128, 128)

→ Conv2D(3 → 32) + ReLU + MaxPool2d
→ Conv2D(32 → 64) + ReLU + MaxPool2d
→ Conv2D(64 → 128) + ReLU + MaxPool2d
→ Conv2D(128 → 256) + ReLU + MaxPool2d
→ Flatten
→ Linear(256*8*8 → 512) + ReLU
→ Linear(512 → 42)
```

* Output: **42 values** representing (x, y) coordinates of 21 keypoints

---

### Training Setup

* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam (learning rate = 1e-4)
* **Batch Size:** 32
* **Early Stopping:** Patience of 5 epochs
* **Best Model Saved as:** `best_hand_model.pth`

---
