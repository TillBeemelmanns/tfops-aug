# Implementation of Google's Auto-Augmentation based on TF2 OPS

Exemplary implementation for learning augmentation policies from your training data distribution. The augmentation
operations rely on Tf2 operations which allows scalability and high computational throughput even on large images. 
Furthermore, the augmentation pipeline can be easily integrated into the tf.data API.

### Example for an augmentation policy
```python
policy = {'sub_policy0': {'op0': ['adjust_saturation', 0.2, 2],
                          'op1': ['equalize', 0.1, 6],
                          'op2': ['add_noise', 0.9, 6]},
          'sub_policy1': {'op0': ['adjust_contrast', 0.1, 7],
                          'op1': ['add_noise', 0.0, 10]},
          'sub_policy2': {'op0': ['posterize', 0.9, 6],
                          'op1': ['unbiased_gamma_sampling', 0.5, 1]},
          'sub_policy3': {'op0': ['adjust_brightness', 0.3, 1],
                          'op1': ['adjust_hue', 0.4, 5]},
          'sub_policy4': {'op0': ['adjust_saturation', 0.2, 9],
                          'op1': ['add_noise', 0.1, 0]},
          'sub_policy5': {'op0': ['adjust_contrast', 1.0, 1],
                          'op1': ['unbiased_gamma_sampling', 0.4, 9]},
          'sub_policy6': {'op0': ['unbiased_gamma_sampling', 0.3, 0],
                          'op1': ['adjust_hue', 0.1, 6]},
          'sub_policy7': {'op0': ['solarize', 0.6, 0],
                          'op1': ['adjust_gamma', 0.3, 6]},
          'sub_policy8': {'op0': ['adjust_jpeg_quality', 0.7, 10],
                          'op1': ['adjust_hue', 0.1, 2]},
          'sub_policy9': {'op0': ['equalize', 0.6, 0],
                          'op1': ['solarize', 0.0, 6]}}
```
Similar to Google's AutoAugment, a single augmentation policy consists of several subpolicies, which inturn consists of one or more 
augmentation operation. Each operation is defined as a tuple of **augmentation method**, 
**probability** and **intensity**. Several operations within one subpolicy are applied in sequence. 
The augmentation policy from above would result in the following:
 
![](assets/augmentation_policy.gif)

### Usage
A full example script for image classification can be found in [classification_example.py](classification_example.py).
This excerpt demonstrates the simplicity for the usage of the augmentation methods:
```python
import tensorflow as tf
from augmentation_operations import apply_augmentation_policy

def augmentor_func(img, label):
    img = apply_augmentation_policy(img, policy)
    return img, label

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(180, 180),
    batch_size=1
).unbatch()

train_dataset = train_dataset.map(augmentor_func).batch(32).prefetch(32)
```



### Augmentation Methods
A list of all implemented augmentation techniques is given here. Additional, methods will be implemented in the near 
future. Performance is measured with the `test_image.jpg` which has size `2048 x 1024`. All augmentation methods are 
executed with `level=5`. Averaged over 500 samples on the Intel Core i7 Prozessor 8665U.

| Augmentation   |      Image      |   Performance (per Image)   |
|----------|:-------------|:-------------:|
| Additive Gaussian Noise | ![](assets/add_noise.gif) | 0.05 s |
| Adjust Brightness | ![](assets/adjust_brightness.gif) | 0.02 s |
| Adjust Contrast | ![](assets/adjust_contrast.gif) | 0.03 s |
| Adjust Gamma | ![](assets/adjust_gamma.gif) | 0.04 s |
| Adjust Hue | ![](assets/adjust_hue.gif) | 0.02 s |
| Adjust JPEG Quality | ![](assets/adjust_jpeg_quality.gif) | 0.05 s |
| Adjust Saturation | ![](assets/adjust_saturation.gif) | 0.02 s |
| Histogramm Equalization | ![](assets/equalize.gif) | 0.05 s |
| Invert | ![](assets/invert.gif) | 0.02 s |
| Posterize | ![](assets/posterize.gif) | 0.02 s |
| Solarize | ![](assets/solarize.gif) | 0.03 s |
| Unbiased Gamma Sampling | ![](assets/unbiased_gamma_sampling.gif) | 0.04 s |
| Gaussian Blur | ![](assets/gaussian_blur.gif) | 0.61 s |
| Sharpen | ![](assets/sharpen.gif) | 0.11 s |
| Shear X | ![](assets/shear_x.gif) | 0.06 s |
| Shear Y | ![](assets/shear_y.gif) | 0.06 s |
| Translate X | ![](assets/translate_x.gif) | 0.09 s |
| Translate Y | ![](assets/translate_y.gif) | 0.09 s |

### TODO
- [ ] More Augmentation Methods
    - [X] Shear X
    - [X] Shear Y
    - [X] Translate X
    - [X] Translate Y
    - [ ] Random Translation
    - [ ] Random Rotation
- [ ] Implement Learning Pipeline
- [ ] Implement augmentation policy search with Ray Tune
- [ ] Clean up Code (Unified Docstrings)
