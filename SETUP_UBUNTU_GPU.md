# Ubuntu + RTX 3060 setup for training

Run each block in order on the Ubuntu machine. Stop at the first block that fails and fix it before moving on.

## 1. Is the GPU visible to the OS?

```bash
lspci | grep -i nvidia
```
Expected: a line containing `GeForce RTX 3060`. If nothing prints, the card is not seated / powered — check hardware first.

## 2. Is the NVIDIA driver installed?

```bash
nvidia-smi
```
Expected: a table showing `NVIDIA GeForce RTX 3060`, driver version (>= 535) and CUDA version (>= 12.x).

If `command not found` or an error, install the driver:

```bash
sudo ubuntu-drivers autoinstall && sudo reboot
```
(After reboot re-run `nvidia-smi`.) Alternative if autoinstall picks nothing:
`sudo apt install nvidia-driver-550 && sudo reboot`.

## 3. Python environment (TensorFlow with bundled CUDA)

TensorFlow >= 2.16 ships its own CUDA/cuDNN libraries via pip, so you do **not**
need to install the CUDA toolkit manually - only the driver from step 2.

```bash
sudo apt install -y python3-venv python3-pip git
```

```bash
cd ~ && python3 -m venv drenv && source drenv/bin/activate && pip install --upgrade pip
```

```bash
pip install "tensorflow[and-cuda]==2.16.1" numpy pandas scikit-learn opencv-python-headless matplotlib seaborn tqdm jupyter nbformat gradio
```

## 4. Verify TensorFlow sees the GPU

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```
Expected last line: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

If it prints `[]`:
- `nvidia-smi` must work first (step 2).
- Try: `export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH` then re-run.
- Python must be 3.9-3.12 for TF 2.16 (`python3 --version`).

## 5. Project + data on the Ubuntu box

Copy the project folder and the raw dataset so the layout is:

```
<somewhere>/
  archive/                         <- raw Kaggle data (2.1 GB)
    trainLabels.csv
    colored_images/colored_images/{No_DR,Mild,Moderate,Severe,Proliferate_DR}/
  dr_project/                      <- this repo
```

Options: USB drive, `rsync`/`scp` over LAN, or re-download `archive/` on Ubuntu with
the Kaggle CLI (`kaggle datasets download -d sovitrath/diabetic-retinopathy-2015-data-colored-resized`).

`outputs/processed/` (2.7 GB) does **not** need copying - re-run notebooks 01 and 02 on
Ubuntu (~5 min) and it is regenerated identically (pipeline is deterministic, seed fixed).

## 6. Run notebooks headless from the terminal

```bash
cd <somewhere>/dr_project/notebooks && source ~/drenv/bin/activate && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_eda.ipynb 02_preprocessing.ipynb
```

Or open them in JupyterLab: `jupyter lab` and run top-to-bottom.

## 7. Quick GPU sanity benchmark (optional)

```bash
python -c "import tensorflow as tf, time; x=tf.random.normal((4096,4096)); t=time.time(); [tf.matmul(x,x) for _ in range(20)]; print('20 matmuls:', round(time.time()-t,2),'s (GPU should be < 1 s)')"
```
