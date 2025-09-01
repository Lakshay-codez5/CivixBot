import opendatasets as od
import pathlib

# Civic Issues Dataset (small, ~30MB)
dataset_url = "https://www.kaggle.com/datasets/shubhendra19/civic-issues-image-dataset"

# Download into 'datasets/civic_issues'
import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/shubhendra19/civic-issues-image-dataset",
    data_dir="datasets/civic_issues",
    kaggle_username="gauravsharma56",
    kaggle_key="dfe4b6f9d8143e06ddf1b767280f5c67"
)


# Convert path for TensorFlow / pathlib
data_dir = pathlib.Path("datasets/civic_issues/civic-issues-image-dataset")
print("Dataset downloaded to:", data_dir)
