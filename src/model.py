"""
Model definitions: a small from-scratch CNN (baseline) and ImageNet
transfer-learning models (EfficientNet / ResNet).

Every model takes float32 RGB images in [0, 255] of shape
(IMG_SIZE, IMG_SIZE, 3) and outputs 5 softmax probabilities. Backbone-
specific normalisation lives *inside* the model so the saved .keras file is
self-contained: the Gradio app just feeds it the preprocessed image.
"""
import keras
from keras import layers

from . import config

# name -> (constructor, input normalisation the pretrained weights expect)
BACKBONES = {
    "EfficientNetB0": (keras.applications.EfficientNetB0, "builtin"),
    "EfficientNetB3": (keras.applications.EfficientNetB3, "builtin"),
    "ResNet50":       (keras.applications.ResNet50,       "caffe"),
    "MobileNetV2":    (keras.applications.MobileNetV2,    "tf"),
}


@keras.saving.register_keras_serializable(package="dr")
class ImageNetNormalise(layers.Layer):
    """
    Reproduce `keras.applications.<backbone>.preprocess_input` as a layer.

    mode "builtin" - identity (EfficientNet has Rescaling+Normalization inside)
    mode "caffe"   - RGB->BGR, subtract ImageNet mean per channel  (ResNet50)
    mode "tf"      - scale to [-1, 1]                              (MobileNetV2)
    """
    def __init__(self, mode: str = "builtin", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, x):
        if self.mode == "caffe":
            x = x[..., ::-1]
            return x - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)
        if self.mode == "tf":
            return x / 127.5 - 1.0
        return x

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


# ---------------------------------------------------------------------------
# Baseline: small CNN trained from scratch
# ---------------------------------------------------------------------------
def build_baseline_cnn(img_size: int = config.IMG_SIZE, dropout: float = 0.3,
                       filters=(32, 64, 128, 256)) -> keras.Model:
    """
    4-block VGG-style CNN (~1.2 M params). Exists to show how much
    ImageNet pre-training helps compared with learning from 25 k images.
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    for i, f in enumerate(filters):
        for j in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"b{i}_conv{j}")(x)
            x = layers.BatchNormalization(name=f"b{i}_bn{j}")(x)
            x = layers.Activation("relu", name=f"b{i}_relu{j}")(x)
        x = layers.MaxPooling2D(name=f"b{i}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax",
                           dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


# ---------------------------------------------------------------------------
# Transfer learning
# ---------------------------------------------------------------------------
def build_transfer_model(backbone: str = "EfficientNetB0", dropout: float = 0.3,
                         img_size: int = config.IMG_SIZE) -> keras.Model:
    """
    ImageNet backbone (frozen) + GlobalAveragePooling + Dropout + Dense(5).

    The backbone is returned frozen (stage A: train the head only). Call
    `unfreeze_top` before stage B fine-tuning. The nested backbone model is
    reachable via `get_backbone(model)`.
    """
    ctor, norm = BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet",
                input_shape=(img_size, img_size, 3))
    base.trainable = False

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = ImageNetNormalise(norm, name="normalise")(inputs)
    x = base(x, training=False)          # keep BatchNorm stats frozen in stage A
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax",
                           dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_dr")


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested pretrained backbone (the only sub-model in the graph)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("model has no nested backbone (is it the baseline CNN?)")


def unfreeze_top(model: keras.Model, fraction: float = 0.3,
                 keep_bn_frozen: bool = True) -> int:
    """
    Make the top `fraction` of backbone layers trainable for fine-tuning.

    Lower layers (edges, blobs) transfer as-is; the top layers are adapted
    to retinal texture. BatchNorm layers stay frozen: with small batches
    their running statistics would drift and wreck the pretrained features.
    Returns the number of layers that were unfrozen. Re-compile afterwards.
    """
    base = get_backbone(model)
    base.trainable = True
    n = len(base.layers)
    cut = int(n * (1 - fraction))
    unfrozen = 0
    for i, layer in enumerate(base.layers):
        if i < cut or (keep_bn_frozen and isinstance(layer, layers.BatchNormalization)):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    return unfrozen


def compile_model(model: keras.Model, lr: float,
                  label_smoothing: float = config.LABEL_SMOOTHING,
                  weight_decay: float = config.WEIGHT_DECAY) -> keras.Model:
    """
    AdamW + label-smoothed categorical cross-entropy.
    Metrics: accuracy and macro-F1 (the imbalance-aware number we select on).
    """
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc"),
                 keras.metrics.F1Score(average="macro", name="macro_f1")],
    )
    return model


def count_params(model: keras.Model) -> dict:
    """Total / trainable parameter counts in millions."""
    total = model.count_params()
    trainable = sum(int(keras.ops.size(w)) for w in model.trainable_weights)
    return {"params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}
