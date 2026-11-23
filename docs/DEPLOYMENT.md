# Deployment: web API + browser UI

The app loads a **model bundle** exported at the end of the tuning notebook (`deploy_bundle.zip`) and serves:

- a browser page (`/`): upload an ultrasound image, see the lesion outline, the segmentation probability map,
  the probability of malignancy against the decision threshold, and a decision text;
- a JSON API (`/api/predict`) for other programs.

CPU is enough: one image takes about 1–4 s, depending on the number of models in the ensemble and on TTA.

> Research prototype. Not a medical device. Not for diagnosis or treatment decisions.

## The bundle

```
deploy_bundle/
├── bundle.json   threshold, post-processing, TTA, model list, reference metrics, library versions
├── seg_0.pt …    segmentation models (float16 weights + their config)
└── cls_0.pt …    classification models
```
`python -m bus.bundle info deploy_bundle` prints the settings and the stored test metrics.
The bundle is not in git (too large); keep `deploy_bundle.zip` from Kaggle.

## Option A: Hugging Face Spaces (free public URL, nothing installed on your PC)

1. Create a free account at huggingface.co → **New Space** → name it (e.g. `breast-ultrasound-demo`),
   **SDK: Docker** (blank template), hardware **CPU basic (free)**. Choose *Private* if only you (and your
   examiners, invited) should see it.
2. In the Space: **Files → Add file → Upload files**, and upload from `E:\MTP`:
   - `Dockerfile`, `requirements-app.txt`
   - the folders `app/` and `src/` (drag the folders in)
   - the folder `deploy_bundle/` (unzip `deploy_bundle.zip` first; the `.pt` files are stored with Git LFS
     automatically)
   - `deploy/hf_space_README.md`, **renamed to `README.md`** (its header tells the Space to use Docker on port 7860)
3. Commit. The Space builds the image (5–10 min, log under *Logs*) and then shows the app at
   `https://huggingface.co/spaces/<user>/<space>`.
4. To update the models later, upload a new `deploy_bundle/`; the Space rebuilds by itself.

## Option B: any computer or server with Docker

```bash
# in the repository root, with the bundle unzipped to ./deploy_bundle
docker build -t bus-app .
docker run -p 7860:7860 bus-app
# open http://localhost:7860
```

## Option C: plain Python (any machine, no Docker)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
uvicorn app.server:app --host 0.0.0.0 --port 7860      # from the repository root
```
Environment variables: `BUS_BUNDLE` (bundle folder, default `./deploy_bundle`), `BUS_DEVICE` (`cpu` / `cuda`).

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok", "models_loaded": true, ...}` |
| GET | `/api/info` | threshold, models, TTA, reference metrics, disclaimer |
| POST | `/api/predict` | form field `file` = image (PNG / JPG / BMP / TIFF, max 20 MB) |

Example:
```bash
curl -F "file=@scan.png" http://localhost:7860/api/predict
```
Response fields: `lesion_found`, `prob_malignant` (null if no lesion), `threshold`, `decision`,
`level` (`high` / `low` / `none`), `lesion_box` `[x0, y0, x1, y1]`, `lesion_area_pct`, `image_size`,
`overlay` and `heatmap` (PNG data URLs), `elapsed_ms`, `model`, `disclaimer`.

Interactive API documentation is generated automatically at `/docs`.

## How a prediction is made

1. The image is read as grayscale; very large images are downscaled to 1,600 px on the long side.
2. Each segmentation model predicts a lesion probability map (and, with TTA, on the mirrored image too);
   the maps are averaged and thresholded at 0.5; optionally only the largest region is kept.
3. No region → "No lesion detected" (no malignancy score is shown: the classifier was trained on lesions only).
4. Otherwise each classifier looks at its crop around the lesion (context as configured) and the
   probabilities are averaged.
5. Probability ≥ the bundle threshold (chosen on validation data for 90 % sensitivity) → "Suspicious for
   malignancy"; otherwise "Lesion with benign-looking features".

## Checked on Kaggle

The last section of `notebooks/kaggle_tuning.ipynb` loads the exported bundle, runs it on a BUS-BRA image, and
sends one request through FastAPI's test client to `/api/predict` and `/`, so the app code is verified before
you download the bundle.
