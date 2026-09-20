"""
Gradio demo: upload a fundus photograph -> preprocessing preview, predicted
DR stage with class probabilities, referral advice and a Grad-CAM heat-map.

Run:  python app/app.py            (from the dr_project folder)
Needs outputs/models/<final_run>.keras and outputs/metrics/<final_run>.json
(both produced by notebook 04). On Kaggle, run with share=True to get a public link.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np

from src import config
from src import evaluate as E
from src import preprocess as pp

# ---------------------------------------------------------------------------
STAGE_INFO = {
    0: ("No DR", "No signs of diabetic retinopathy.", "Routine re-screening in 12 months."),
    1: ("Mild NPDR", "Microaneurysms only.", "Re-screen in 6-12 months."),
    2: ("Moderate NPDR", "Microaneurysms, haemorrhages, hard exudates.", "Refer to ophthalmology (referable DR)."),
    3: ("Severe NPDR", "Extensive haemorrhages, venous beading, IRMA.", "Urgent referral."),
    4: ("Proliferative DR", "Neovascularisation, vitreous haemorrhage.", "Urgent referral - vision-threatening."),
}

print("loading model ...")
MODEL, CFG = E.load_model_fp32()
CAM = E.GradCAM(MODEL)
print(f"loaded {CFG['run']} ({CFG['backbone']}, val QWK {CFG['val_qwk']:.3f})")


def predict(image: np.ndarray, show_cam: bool = True):
    """Gradio callback. image: RGB uint8 array from the upload widget."""
    if image is None:
        return None, None, "Upload a fundus image.", None
    proc = pp.preprocess(image, size=CFG["img_size"])          # same pipeline as training
    x = proc.astype("float32")
    probs = MODEL.predict(x[None], verbose=0)[0]
    stage = int(probs.argmax())
    name, signs, advice = STAGE_INFO[stage]
    p_ref = float(probs[2:].sum())

    verdict = (f"## {name}  (stage {stage})\n"
               f"**Confidence:** {probs[stage]:.1%}\n\n"
               f"**Typical signs:** {signs}\n\n"
               f"**Referral advice:** {advice}\n\n"
               f"P(referable DR, stage >= Moderate) = **{p_ref:.1%}**\n\n"
               f"_Screening aid only - not a diagnosis. Model: {CFG['backbone']} "
               f"(val QWK {CFG['val_qwk']:.2f})._")
    prob_dict = {config.CLASS_NAMES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)}

    cam_img = None
    if show_cam:
        heat, _, _ = CAM(x, class_index=stage)
        cam_img = E.overlay(proc, heat)
    return proc, prob_dict, verdict, cam_img


with gr.Blocks(title="Diabetic Retinopathy Stage Detection") as demo:
    gr.Markdown("# Diabetic Retinopathy Stage Detection\n"
                "Upload a colour fundus photograph. The image is preprocessed (crop, CLAHE, "
                "denoise, sharpen), classified into one of five DR stages by a fine-tuned "
                f"**{CFG['backbone']}**, and explained with a Grad-CAM heat-map.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Fundus image (upload)")
            cam_toggle = gr.Checkbox(value=True, label="Show Grad-CAM")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=1):
            out_proc = gr.Image(label="Preprocessed input (what the model sees)")
            out_cam = gr.Image(label="Grad-CAM (where the model looked)")
    with gr.Row():
        out_probs = gr.Label(num_top_classes=5, label="Stage probabilities")
        out_text = gr.Markdown()
    btn.click(predict, [inp, cam_toggle], [out_proc, out_probs, out_text, out_cam])
    inp.upload(predict, [inp, cam_toggle], [out_proc, out_probs, out_text, out_cam])

    # A few example images from the test split so the demo works with one click
    try:
        from src import data
        _, _, test_df = data.load_splits()
        ex = [[test_df[test_df.label == c].iloc[0].image_path] for c in range(config.NUM_CLASSES)
              if Path(test_df[test_df.label == c].iloc[0].image_path).exists()]
        if ex:
            gr.Examples(ex, inputs=inp, label="Examples (one per stage: 0-4)")
    except Exception as e:                       # dataset not present -> no examples
        print("no examples:", e)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
