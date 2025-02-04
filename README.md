# U-of-Wyo-ML-5555-01-22772-Spring-2023-Research-Project

Here we explore the effect of outpainted training data on CIFAR-10 image classifier performance. Spoiler alert: it didn't help very much. The [report](https://github.com/Mikey-E/U-of-Wyo-COSC-5555-01-22772-Spring-2023-Research-Project/blob/main/Report/latex/project_report.pdf) is the main thing worth looking at here, if you're interested. Trying to run the code probably won't work since not all the data is available anymore.

### Dependency Versions

- keras 2.12.0
- tensorflow 2.12.0
- numpy 1.23.5
- matplotlib 3.7.0

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

### Presentation

https://docs.google.com/presentation/d/1M0CH5_qJHTCSna_k0GoQurjjvSQ2R3Grq4Bn3dNkPno/edit

### Models

At least one model (the size 64 one) is too big to push to Github. If you wish, you can of course make it yourself
from scratch using the training data. Simply uncomment the 2 training lines in the main function in classifier.py
