# Implementation of Google's Auto-Augmentation based on TF2 Operations

Exemplary implementation for learning augmentation policies from your data distribution. The augmentation operations 
rely solely on tf operations which allows scalability and high computational throughput even on large images.
 
Example for an augmentation policy.
```python
augmentation_policy = { 'sub_policy0': {'op1': ['adjust_saturation', 1, 2],
                                        'op2': ['equalize', 1, 6]},
                        'sub_policy1': {'op1': ['adjust_contrast', 1, 7],
                                        'op2': ['add_noise', 1, 10]},
                        'sub_policy2': {'op1': ['posterize', 1, 6],
                                        'op2': ['unbiased_gamma_sampling', 1, 1]},
                        'sub_policy3': {'op1': ['adjust_brightness', 1, 1],
                                        'op2': ['adjust_hue', 1, 5]},
                        'sub_policy4': {'op1': ['adjust_saturation', 1, 9],
                                        'op2': ['add_noise', 1, 0]},
                        'sub_policy5': {'op1': ['adjust_contrast', 1, 1],
                                        'op2': ['unbiased_gamma_sampling', 1, 9]},
                        'sub_policy6': {'op1': ['unbiased_gamma_sampling', 1, 0],
                                        'op2': ['adjust_hue', 1, 6]},
                        'sub_policy7': {'op1': ['solarize', 1, 0],
                                        'op2': ['adjust_gamma', 1, 6]},
                        'sub_policy8': {'op1': ['adjust_jpeg_quality', 1, 10],
                                        'op2': ['adjust_hue', 1, 2]},
                        'sub_policy9':{'op1': ['equalize', 1, 0],
                                       'op2': ['solarize', 1, 6]}}
```