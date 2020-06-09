# WIP: Implementation of Google's Auto-Augmentation based on TF2 Operations

Exemplary implementation for learning augmentation policies from your data distribution. The augmentation operations 
rely solely on tf operations which allows scalability and high computational throughput even on large images.
 
tf.data API

Example for an augmentation policy.

```python
augmentation_policy = {'sub_policy0': {'op0': ['adjust_saturation', 0.2, 2],
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


| Augmentation   |      Image      |
|----------|:-------------:|
| Additive Gaussian Noise | ![](assets/add_noise.gif) |
| Adjust Brightness | ![](assets/adjust_brightness.gif) |
| Adjust Contrast | ![](assets/adjust_contrast.gif) |
| Adjust Gamma | ![](assets/adjust_gamma.gif) |
| Adjust Hue | ![](assets/adjust_hue.gif) |
| Adjust JPEG Quality | ![](assets/adjust_jpeg_quality.gif) |
| Adjust Saturation | ![](assets/adjust_saturation.gif) |
| Histogramm Equalization | ![](assets/equalize.gif) |
| Invert | ![](assets/invert.gif) |
| Posterize | ![](assets/posterize.gif) |
| Solarize | ![](assets/solarize.gif) |
| Unbiased Gamma Sampling | ![](assets/unbiased_gamma_sampling.gif) |

