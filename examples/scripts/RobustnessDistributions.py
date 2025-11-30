# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib.util
import logging
from pathlib import Path
import random
import argparse
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import ada_verona.util.logger as logger
from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.attacks.pgd_attack import PGDAttack
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator

logger.setup_logging(level=logging.INFO)

torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description='Robustness distribution script')
    parser.add_argument('--directory_path', type=str, default='"examples/CNNYangBig/emnist_cnn_yang_big-pgd-training_21-10-2025+13_06"', help='Source model path')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'EMNIST', 'CIFAR10', 'CIFAR100'], default='EMNIST', help='Dataset to use (MNIST, EMNIST, CIFAR10, CIFAR100)')
    args = parser.parse_args()

    # Load dataset based on argument
    dataset_size = 100
    if args.dataset == 'MNIST':
        torch_dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    elif args.dataset == 'EMNIST':
        torch_dataset = torchvision.datasets.EMNIST('../data', split="balanced", train=True, download=True, transform=torchvision.transforms.ToTensor())
    elif args.dataset == 'CIFAR10':
        # cifar10_mean = [0.4914, 0.4822, 0.4465]
        # cifar10_std = [0.2470, 0.2435, 0.2616]
        # mean_std = np.mean(cifar10_std)
        # normalize = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        cifar_transform = transforms.Compose([
            transforms.ToTensor()        ])
        torch_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=cifar_transform)
    elif args.dataset == 'CIFAR100':
        # cifar100_mean = [0.5071, 0.4865, 0.4409]
        # cifar100_std = [0.2673, 0.2564, 0.2762]
        # mean_std = np.mean(cifar100_std)
        # normalize = transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        cifar_transform = transforms.Compose([
            transforms.ToTensor()        
        ])
        torch_dataset = torchvision.datasets.CIFAR100('../data', train=True, download=True, transform=cifar_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    experiment_name = "pgd"
    timeout = 600
    epsilon_list = list(np.arange(0.001, 24/255, 0.001))
    experiment_repository_path = Path(args.directory_path) / "results" # Path("../CNNYangBig/results")
    network_folder = Path(args.directory_path)

    # Create random indices for sampling
    total_dataset_size = len(torch_dataset)
    subset_indices = random.sample(range(total_dataset_size), dataset_size)
    subset_torch_dataset = torch.utils.data.Subset(torch_dataset, subset_indices)

    dataset = PytorchExperimentDataset(dataset=subset_torch_dataset)

    file_database = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    file_database.initialize_new_experiment(experiment_name)

    file_database.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=args.dataset,
            dataset_details=str(dataset),
            timeout=timeout,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )

    property_generator = One2AnyPropertyGenerator()
    verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=40))

    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)

    network_list = sorted(file_database.get_network_list(), key=lambda network: network.name)

    for network in network_list:
        sampled_data = dataset_sampler.sample(network, dataset)

        for data_point in sampled_data:
            verification_context = file_database.create_verification_context(network, data_point, property_generator)

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            file_database.save_result(epsilon_value_result)

    file_database.save_plots()

    # Copy saved results to experiment_repository_path
    src_results_path = file_database.get_results_path()
    dst_results_path = experiment_repository_path
    
    # Create destination directory if it doesn't exist
    dst_results_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from source results to destination
    for item in src_results_path.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_results_path / item.name)
        elif item.is_dir():
            shutil.copytree(item, dst_results_path / item.name, dirs_exist_ok=True)

if __name__ == "__main__":
    main()