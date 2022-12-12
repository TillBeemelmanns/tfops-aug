augmentation_policy = {
    'sub_policy0': {'op0': ['adjust_saturation', 0.2, 2],
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
                    'op1': ['solarize', 0.0, 6]}
}

classification_policy = {
    'sub_policy0': {'op0': ['adjust_saturation', 0.2, 2],
                    'op1': ['equalize', 0.5, 6]},
    'sub_policy1': {'op0': ['translate_y', 0.5, 7],
                    'op1': ['shear_x', 0.3, 6]},
    'sub_policy2': {'op0': ['posterize', 0.9, 6],
                    'op1': ['unbiased_gamma_sampling', 0.5, 1]},
    'sub_policy3': {'op0': ['equalize', 0.3, 1],
                    'op1': ['adjust_hue', 0.4, 5]},
    'sub_policy4': {'op0': ['adjust_saturation', 0.2, 9],
                    'op1': ['shear_y', 0.5, 5]},
    'sub_policy5': {'op0': ['adjust_contrast', 1.0, 1],
                    'op1': ['unbiased_gamma_sampling', 0.4, 9]},
    'sub_policy6': {'op0': ['unbiased_gamma_sampling', 0.3, 0],
                    'op1': ['adjust_hue', 0.1, 6]},
    'sub_policy7': {'op0': ['solarize', 0.6, 0],
                    'op1': ['adjust_gamma', 0.3, 6]},
    'sub_policy8': {'op0': ['adjust_jpeg_quality', 0.7, 10],
                    'op1': ['adjust_hue', 0.1, 2]},
    'sub_policy9': {'op0': ['equalize', 0.6, 0],
                    'op1': ['shear_x', 0.5, 6]}
}

test_policy = {
    'sub_policy0': {'op0': ['adjust_saturation', 1.0, 2],
                    'op1': ['equalize', 1.0, 6],
                    'op2': ['add_noise', 1.0, 6]},
    'sub_policy1': {'op0': ['adjust_contrast', 1.0, 7]},
    'sub_policy2': {'op0': ['posterize', 1.0, 6],
                    'op1': ['unbiased_gamma_sampling', 1.0, 1]},
    'sub_policy3': {'op0': ['adjust_brightness', 1.0, 1],
                    'op1': ['adjust_hue', 1.0, 5]},
    'sub_policy4': {'op0': ['adjust_saturation', 0.2, 9],
                    'op1': ['add_noise', 1.0, 0]},
    'sub_policy5': {'op0': ['adjust_contrast', 1.0, 1],
                    'op1': ['unbiased_gamma_sampling', 1.0, 9]},
    'sub_policy6': {'op0': ['unbiased_gamma_sampling', 1.0, 0],
                    'op1': ['adjust_hue', 1.0, 6]},
    'sub_policy7': {'op0': ['solarize', 1.0, 0],
                    'op1': ['adjust_gamma', 1.0, 6]},
    'sub_policy8': {'op0': ['adjust_jpeg_quality', 1.0, 10],
                    'op1': ['adjust_hue', 1.0, 2]},
    'sub_policy9': {'op0': ['equalize', 1.0, 0],
                    'op1': ['solarize', 1.0, 6]}
}


ALL_POLICIES = [augmentation_policy, classification_policy, test_policy]