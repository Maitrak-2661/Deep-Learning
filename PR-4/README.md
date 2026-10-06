# Project B — Teachable Machine

**Subject:** Deep Learning · Practical Report 4 (PR 4)
**Declaration:** I have chosen **Project B — Teachable Machine** (Project A was not attempted).
**Name:** _Your Name_ · **GRID:** _Your GRID_

Two models were trained visually on [Teachable Machine](https://teachablemachine.withgoogle.com), exported, and deployed:

| # | Model | Problem | Where it runs |
|---|---|---|---|
| 1 | Image | Waste Sorting Assistant (10 classes) | Python / OpenCV webcam app in a Jupyter notebook |
| 2 | Pose | Posture classifier (4 classes) | Live in the browser (`index.html`), model inspected in Python |

> The audio model is not part of this submission.

---

## Repository structure

```
PR-4/
├── README.md
├── CONCEPTS.md
├── garbage-segmentation/            # Image model
│   ├── DL_PR4B_image.ipynb
│   ├── requirements.txt
│   ├── models/image/                # keras_model.h5, labels.txt
│   ├── output/                      # waste_log.csv, b1_image_result.jpg
│   └── screenshots/                 # Teachable Machine training screenshot
└── human pose dataset/              # Pose model
    ├── DL_PR4B_pose.ipynb
    ├── index.html                   # live browser demo
    ├── requirements.txt
    ├── models/pose/                 # model.json, metadata.json, weights.bin
    ├── output/                      # demo screenshot
    └── screenshots/                 # Teachable Machine training screenshot
```

---

## Datasets

| Model | Dataset | Source |
|---|---|---|
| Image | Garbage Classification V2 | https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2 |
| Pose | Silhouettes for Human Posture Recognition | https://www.kaggle.com/datasets/mexwell/silhouettes-for-human-posture-recognition |

Images from these datasets were uploaded to Teachable Machine class by class to train each model.

---

## Model 1 — Image: Waste Sorting Assistant

**Problem:** Identify the type of waste shown to the camera and tell the user which bin it belongs in.

| Item | Details |
|---|---|
| Classes (10) | trash, plastic, clothes, shoes, paper, cardboard, biological, battery, glass, metal |
| Samples per class | _fill in from your Teachable Machine screenshot_ |
| Export | TensorFlow → Keras (`keras_model.h5` + `labels.txt`) |
| Input | 224 × 224 × 3 image, normalised to [-1, 1] |
| Output | 10 softmax probabilities |
| Architecture | MobileNet feature extractor (410,208 params) + Dense head (129,100 params), 539,308 total |

**Application:** the notebook opens the webcam, shows the predicted class with a confidence bar, and maps it to a bin:

| Bin | Items |
|---|---|
| Recyclable | plastic, paper, cardboard, glass, metal |
| Organic | biological |
| Hazardous | battery |
| General | trash, clothes, shoes |

Every 5 seconds the detection is appended to `garbage-segmentation/output/waste_log.csv` (time, class, confidence, bin).

**Controls:** `s` saves a screenshot, `q` quits.



## Model 2 — Pose: Posture Classifier

**Problem:** Classify body posture from the webcam.

| Item | Details |
|---|---|
| Classes (4) | bending, lying, sitting, standing |
| Samples per class | _fill in from your Teachable Machine screenshot_ |
| Export | TensorFlow.js (`model.json`, `metadata.json`, `weights.bin`) |
| Feature extractor | PoseNet (MobileNetV1, output stride 16, input 257, multiplier 0.75) |
| Classifier head | Dense(100, relu) → Dropout(0.5) → Dense(4, softmax) |
| Input | 14,739-value PoseNet output vector |
| Parameters | 1,474,400 (all in the trained head) |

**Why the live demo is in the browser:** Teachable Machine pose projects can only be exported as TensorFlow.js. The model's input is the raw output of PoseNet running inside the Teachable Machine JavaScript library, not an image and not 17 keypoints, so it cannot be fed directly from Python or MoveNet. The live classification therefore runs in `index.html`. `DL_PR4B_pose.ipynb` rebuilds the same Dense head in Keras from `weights.bin` and prints the model summary and parameter counts.



## Setup and how to run

Python 3.10+ is recommended.

### Image model
```bash
cd garbage-segmentation
pip install -r requirements.txt
jupyter notebook
```
Open `DL_PR4B_image.ipynb` and run the cells from the top. The notebook must be run from inside the `garbage-segmentation` folder so that `models/image/keras_model.h5` is found.

### Pose model
```bash
cd "human pose dataset"
pip install -r requirements.txt
jupyter notebook          # DL_PR4B_pose.ipynb: inspect the model

python -m http.server 8000
```
Then open http://localhost:8000 in Chrome, click **Start** and allow the camera. Do not open `index.html` by double-clicking, because the model and webcam will not load from a `file://` page. An internet connection is needed, since the TensorFlow.js and PoseNet libraries load from a CDN.

> Note: Teachable Machine `.h5` files fail to load with `tf.keras` on recent TensorFlow versions (`DepthwiseConv2D` error). The notebooks use the `tf_keras` package for this reason.

---

## How Teachable Machine works (summary)

1. Teachable Machine trains a small Dense classification head on top of a frozen, pre-trained feature extractor: MobileNet for images, PoseNet for poses.
2. This is **Transfer Learning via Feature Extraction**. The base network was pre-trained on a large dataset, so it already knows edges, textures and shapes. Only the final head is trained on our own classes, which is why a small dataset is enough.
3. It is the same idea as manual Feature Extraction with `base_model.trainable = False` in Keras, done visually in the browser.
4. Preprocessing must match training: the image model expects 224 × 224 input normalised to [-1, 1].
5. The pose model classifies body-part positions, not pixels, so it is less sensitive to lighting and clothing, but it needs the full body in frame.
6. **Limitation:** the models are small and tuned for speed. For higher accuracy, a larger network could be fine-tuned.

Full notes and model introspection table: [CONCEPTS.md](CONCEPTS.md)

---

## Model introspection

| Model | Input | Output | Top-level layers | Total params |
|---|---|---|---|---|
| Image (waste) | (None, 224, 224, 3) | (None, 10) | 2 (MobileNet base + Dense head) | 539,308 |
| Pose (posture) | (None, 14739) | (None, 4) | 3 (Dense, Dropout, Dense) | 1,474,400 |

---

## Limitations and scope

- The audio model and the combined application were not built.
- The pose model is deployed in the browser, not in a Python script, because of the export format described above.
- Accuracy depends on camera angle, lighting and background; the models were trained on dataset images and tested with a live webcam.

---

## Video

[Watch the demo video](PASTE_YOUR_VIDEO_LINK_HERE)

---

## Tools used

TensorFlow, tf_keras, OpenCV, NumPy, Matplotlib, Pandas, Jupyter, Teachable Machine, PoseNet (TensorFlow.js), HTML / JavaScript
