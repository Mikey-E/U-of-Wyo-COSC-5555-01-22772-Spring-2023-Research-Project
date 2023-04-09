# U-of-Wyo-ML-5555-01-22772-Spring-2023-Research-Project

### Data and Labels

The classifier(s) are trained on only the horses and ships from cifar-10. Horses are labeled 0; ships 1.

### Data Preprocessing

data_preprocess.py contains (for reference) the code used to extract the relevant cifar10 data. There is no need for anyone to
run this to replicate the original cifar10 subset we used. It is included for reproducibility without the burden to
actually run it.

### Outpainting specifications

| Parameter                | Value                              |
|--------------------------|------------------------------------|
| Checkpoint               | v2-1_768-ema-pruned.ckpt [ad2a33c361] |
| Resize mode              | Just resize                        |
| Sampling method          | Euler a                            |
| Sampling steps           | 20                                 |
| Width                    | 64                                 |
| Height                   | 64                                 |
| CFG scale (no prompt)    | 7                                  |
| Denoising strength       | 0                                  |
| Seed                     | 0                                  |
| Script                   | Outpainting mk2                    |
| Pixels to expand         | 8                                  |
| Mask Blur                | 1                                  |
| Outpainting Directions   | left, right, up, down              |
| Fall-off exponent        | 1                                  |
| Color variation          | 0.05                               |


### Functions to Load Data

There are two functions for loading the datasets: load_data() and load_labels(). They return ordered sets in sync with each
other i.e. the index of a dataset item from load_data has its label at the same index in the list from load_labels.

### References

https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
