# Demo app

```bash
python app/app.py                 # http://localhost:7860
GRADIO_SHARE=1 python app/app.py  # also prints a public *.gradio.live link (for the video / Kaggle)
```

Needs `outputs/models/<final_run>.keras` + `outputs/metrics/<final_run>.json` from notebook 04
(download from the Kaggle run output if running locally).

Flow: upload fundus image -> `src.preprocess.preprocess` (same pipeline as training)
-> model (float32 rebuild) -> stage + probabilities + referral advice -> Grad-CAM overlay.
