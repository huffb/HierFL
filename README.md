# HierFL

HierFL is a PyTorch implementation of hierarchical federated learning in a `Client -> Edge -> Cloud` topology, plus a small Flask demo for starting training jobs and running handwritten-digit recognition.

## What is included

- `hierfavg.py`: training entrypoint for hierarchical federated averaging
- `datasets/get_data.py`: dataset loading and client split logic
- `App.py`: Flask demo for training, progress polling, and recognition
- `models/`: CNN, logistic regression, and ResNet variants used by the experiments
- `artifacts/`: generated models, metrics, and recognition results

## Environment

- Python 3.10 or newer is recommended
- PyTorch and torchvision must match your local CPU/CUDA environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run CLI training

Minimal example:

```bash
python hierfavg.py --dataset mnist --model lenet --num_clients 10 --num_edges 1 --num_communication 10
```

More complete example:

```bash
python hierfavg.py ^
  --dataset mnist ^
  --model lenet ^
  --num_clients 50 ^
  --num_edges 5 ^
  --frac 1 ^
  --num_local_update 60 ^
  --num_edge_aggregation 1 ^
  --num_communication 100 ^
  --batch_size 20 ^
  --iid 0 ^
  --edgeiid 1 ^
  --show_dis 1 ^
  --lr 0.01 ^
  --lr_decay 0.995 ^
  --lr_decay_epoch 1 ^
  --momentum 0 ^
  --weight_decay 0
```

Training outputs are written to:

- `artifacts/models/`
- `artifacts/metrics/`

## Run the web demo

```bash
python App.py
```

Then open `http://127.0.0.1:5000`.

The web app can:

- start a training job
- poll live training status
- download the latest trained model
- plot accuracy and validation loss curves
- run digit recognition on uploaded images

## Data layout

Datasets are stored under `data/` by default. The current loader supports:

- `mnist`
- `fmnist`
- `cifar10`

You can change the dataset root with `--dataset_root`.

## Configuration split

The project now keeps configuration in two places:

- `options.py`: training arguments and federated-learning defaults
- `config.py`: Flask/Web paths and runtime settings

This keeps experiment settings separate from local app wiring.

## Known notes

- `tensorboardX` is required by `hierfavg.py`
- the web recognition flow currently assumes an MNIST-style model layout
- `fig.py` still contains older standalone plotting logic and has not been folded into the new artifact layout
