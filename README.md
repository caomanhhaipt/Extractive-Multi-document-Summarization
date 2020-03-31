# Extractive Multi-document Summarization using K-means, Centroid-based Method, MMR, and Sentence Position

Python implementation of Extractive Multi-document Summarization as described in [Extractive Multi-document Summarization using K-means, Centroid-based Method, MMR, and Sentence Position](https://dl.acm.org/doi/pdf/10.1145/3368926.3369688)
by Hai Cao Manh, Huong Le Thanh and Tuan Luu Minh.

## Installation

1. Clone this repository.
2. Ensure packages are installed using pip install -r requirements.txt.

## Generate summaries from dataset

```shell
# Running directly from the repository: (path to save: Data/DUC_2007/folder)
methods/main_method/Kmeans_CentroidBase_MMR_SentencePosition.py --folder_to_save="folder"

Notice: if you get a path error, the following command may be helpful:
```shell
# Running directly from the repository:
export PYTHONPATH=.

## Evalute results via rouge

Please replace "test" in the "system_folder" variable with system folder (ex. "folder")
```shell
# Running directly from the repository:
rouge/pyrouge_DUC_2007.py

Notice: if you get an error, you can try running the source code directly with the Pycharm IDE