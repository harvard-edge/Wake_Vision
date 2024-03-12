# Wake Vision
Wake Vision is a Dataset for TinyML person detection. This repository contains the code to generate and filter the dataset from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), as well as code to train and evaluate MobileNetV2 models on the dataset. We also provide a suite of benchmarks to evaluate the performance of a person detection model on challenging subsets.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Download and build Open Images
Instructions in partial_open_images_v7/README.md

## Train a model
To train a MobileNetV2 model using the base config:
```bash
python train.py
```

You can change the config by passing arguments to the train.py script. For example, to change the experiment name and model size, run the following command:
```bash
python train.py --experiment_name="name" --model_size=0.5
```
Alternatively you can change experiment_config.py directly.

## Evaluate the model
To run the benchmark suite:
```bash
python benchmark_suite.py
```
