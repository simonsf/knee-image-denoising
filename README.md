# A Deep Learning-Based De-artifact Diffusion Model for Removing Motion Artifacts in Knee MRI

Our paper, currently under review, was developed by Dr. Yuting Wang's team from the Department of Radiology at the Sichuan Academy of Medical Sciences and Sichuan Provincial People's Hospital in Chengdu, China.

This repository provides the PyTorch implementation of our proposed method, the Conditional De-Motion-Artifact Diffusion Model. Our generative model incorporates the ResNet block and a channel concatenation mechanism to enhance performance.

## Usage
### Environment
```python
pip install -r requirement.txt
```

### Own Data

Download the dataset including 256_256_train/256_256_test/256_256_val
                                ├── hr_256 # ground-truth knee images.
                                └── lr_256 # noised knee images

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/256_256_train", 
        "l_resolution": 256, 
        "r_resolution": 256, 
        "datatype": "npy", 
    },
    "val": {
        "dataroot": "dataset/256_256_test", 
    }
},
```

### Training

```python
# Use train.py and sample.py to train the de-noise task.
# Edit json files to adjust network structure and hyperparameters
python train.py -p train -c config/de-noise.json
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c config/de-noise_infer.json
```

## Acknowledgements

Our work draws inspiration from the following studies:

- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
