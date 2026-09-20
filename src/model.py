"""
Model definitions: pretrained CNN backbones with a small classification head,
plus a from-scratch baseline CNN for comparison.

Transfer-learning recipe (used by train.py):
  phase 1  backbone frozen, train only the head          (high LR)
  phase 2  unfreeze the top blocks, fine-tune everything (low LR, BN frozen)

Input convention: float32 RGB in [0, 255] straight from dataset.py.
Each backbone gets *its own* normalisation layer inside the model so the
saved .keras file is self-contained (no preprocessing to remember at
inference time) and so the pixel statistics always match the ImageNet
weights it was trained with.
"""
import keras
from keras import layers
from keras.applications import (DenseNet121, EfficientNetB0, EfficientNetB3,
                                MobileNetV2, ResNet50V2)

from . import config

# name -> (constructor, normalisation mode)
#   "raw"   backbone contains its own Rescaling (EfficientNet expects 0-255)
#   "tf"    scale to [-1, 1]              (MobileNetV2, ResNet50V2)
#   "torch" /255 then ImageNet mean/std   (DenseNet)
BACKBONES = {
    "efficientnetb0": (EfficientNetB0, "raw"),
    "efficientnetb3": (EfficientNetB3, "raw"),
    "resnet50v2":     (ResNet50V2, "tf"),
    "mobilenetv2":    (MobileNetV2, "tf"),
    "densenet121":    (DenseNet121, "torch"),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalisation(mode: str):
    """Return a list of Keras layers mapping [0,255] -> what the backbone expects."""
    if mode == "raw":
        return []
    if mode == "tf":
        return [layers.Rescaling(1.0 / 127.5, offset=-1.0, name="norm_tf")]
    if mode == "torch":
        return [layers.Rescaling(1.0 / 255.0, name="norm_scale"),
                layers.Normalization(mean=IMAGENET_MEAN,
                                     variance=[s ** 2 for s in IMAGENET_STD],
                                     name="norm_imagenet")]
    raise ValueError(mode)


def build_transfer_model(backbone: str = "efficientnetb0",
                         img_size: int = config.IMG_SIZE,
                         num_classes: int = config.NUM_CLASSES,
                         dropout: float = 0.3,
                         dense_units: int = 256,
                         weights: str = "imagenet") -> keras.Model:
    """
    Pretrained backbone (frozen) + GAP + Dropout + Dense head.

    The classifier is deliberately small: with 24k training images a big
    head would overfit before the backbone adapts.
    """
    ctor, mode = BACKBONES[backbone]
    base = ctor(include_top=False, weights=weights,
                input_shape=(img_size, img_size, 3))
    base.trainable = False                       # phase 1: frozen

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = inputs
    for layer in _normalisation(mode):
        x = layer(x)
    x = base(x, training=False)                  # keep BN in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout / 2, name="drop2")(x)
    # float32 output so mixed-precision training stays numerically safe
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="stage")(x)
    model = keras.Model(inputs, outputs, name=f"dr_{backbone}")
    model.backbone_layer = base                  # handy handle for unfreezing
    return model


def unfreeze_top(model: keras.Model, fraction: float = 0.3) -> int:
    """
    Phase 2: make the top `fraction` of backbone layers trainable.

    BatchNormalization layers stay frozen - with batch size 32 and a domain
    shift from ImageNet, re-estimating BN statistics destabilises training.
    Returns the number of layers unfrozen.
    """
    base = model.backbone_layer
    base.trainable = True
    n = len(base.layers)
    cut = int(n * (1 - fraction))
    unfrozen = 0
    for i, layer in enumerate(base.layers):
        if i < cut or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    return unfrozen


def build_baseline_cnn(img_size: int = config.IMG_SIZE,
                       num_classes: int = config.NUM_CLASSES) -> keras.Model:
    """
    Small CNN trained from scratch. Exists only to show *why* transfer
    learning is needed: same data, same augmentation, no pretrained weights.
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def count_params(model: keras.Model) -> dict:
    """Trainable / non-trainable parameter counts (for the report table)."""
    import numpy as np
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"trainable": tr, "non_trainable": nt, "total": tr + nt}
