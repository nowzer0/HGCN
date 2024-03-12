# HGCN for Detecting Low-Quality Documents
Dataset and code for LREC-COLING 2024 paper "Hierarchical Graph Convolutional Network Approach for Detecting Low-Quality Documents".


## Overview
![HGCN_model_v2](https://github.com/nowzer0/HGCN-for-Detecting-Low-Quality-Documents/assets/80493641/34146650-3842-49a7-a7c0-f61ab397e26c)


## Requirements
* python 3.9.11
* pytorch 2.2.0
* pytorch-lightening 1.2.4
* transformers 4.3.3
* run pip install -r requirements.txt to install rest of the dependencies


## Dataset
Using meta-data to develop a model for detecting low-quality documents with inconsistency issues, we constructed an _Inconsistency Dataset_. 
The data used in the dataset construction covered various news fields and were obtained from one of South Korea’s largest news outlets, Yonhap News.
The dataset contains 216,512 samples split into a 9:0.5:0.5 train-validation-test ratio, resulting in 194,860 training data, 10,826 validation data, and 10,826 test data.

_Inconsistency dataset_ was created by [IDSL](https://sites.google.com/dm.snu.ac.kr/idsl/home) of Hanyang University in Republic of Korea. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1eNIsxaRtHXP6I4VaCiIKmbQRNm_Yo4mT?usp=drive_link).

More information about _Inconsistency dataset_ can be found in Hierarchical Graph Convolutional Network Approach for Detecting Low-Quality Documents.

## Citation
```
@inproceedings{Lee-etal-2024-Hierarchical,
    title = "Hierarchical Graph Convolutional Network Approach for Detecting Low-Quality Documents",
    author = "Lee, Jaeyoung and
      Jang, Joonwon and
      Kim, Misuk",
    booktitle = "The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = May,
    year = "2024",
    address = "Torino, Italia",
    publisher = "International Committee on Computational Linguistics",
    url = "",
    pages = "",
}
```

## License
This code and dataset are licensed under the AFL v3.0 License - see the [License](https://github.com/nowzer0/HGCN-for-Detecting-Low-Quality-Documents/blob/main/LICENSE) file for details.
