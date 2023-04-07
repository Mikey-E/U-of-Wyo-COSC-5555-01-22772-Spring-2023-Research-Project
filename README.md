# U-of-Wyo-ML-5555-01-22772-Spring-2023-Research-Project

### Data Preprocessing

data_preprocess.py contains (for reference) the code used to extract the relevant cifar10 data. There is no need for anyone to
run this to replicate the original cifar10 subset we used. It is included for reproducibility without the burden to
actually run it.

### Load Data

There are two functions for loading the datasets: load_data() and load_labels(). They return ordered sets in sync with each
other i.e. the index of the dataset item from load_data has its label at the same index in the list from load_labels.

### References

https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
