import pytest

from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler





def test_loading_dataset_pytorch():
    return 0 

def test_loading_dataset_from_file():
    return 0

def test loading_dataset_from_file_with_invalid_path():
    dataset = ExperimentDataset()
    with pytest.raises(ValueError):
        dataset.load_dataset_from_file("invalid/path/to/dataset")

def test_sample_loaded_test():


