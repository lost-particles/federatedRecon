import sys

import datasets
from flwr.server.client_proxy import ClientProxy
print(sys.executable)
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from flwr.common import FitIns, EvaluateIns, Parameters
from flwr.server.client_manager import ClientManager
from typing import Dict, List, Tuple
from datetime import datetime
import json
import os
import random
from pathlib import Path

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
disable_progress_bar()

DATASET = "flwrlabs/celeba"
NUM_CLIENTS = 2
BATCH_SIZE = 64
CLIENT_EPOCHS = 30
GAN_EPOCHS = 50000
NUM_CLASSES = 1000  # assuming 1000 unique identities
START_GAN_TRAINING = 5
NUM_ROUNDS = 40
IDENTITIES_PER_CLIENT = 500
TOTAL_IDENTITIES_USED = NUM_CLIENTS * IDENTITIES_PER_CLIENT  # 1000
TARGET_CLASS = 612
TARGET_DETAILS_LOGGING = True



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join("Biometric_Leakage_Outputs", timestamp)
os.makedirs(RUN_DIR, exist_ok=True)
TB_DIR = os.path.join(RUN_DIR, "tensorboard")
os.makedirs(TB_DIR, exist_ok=True)
GAN_MODELS_DIR = os.path.join(RUN_DIR, "GAN_Models")
os.makedirs(GAN_MODELS_DIR, exist_ok=True)


def load_datasets(partition_id: int, return_target_samples: bool = False, target_class: int = None):
    # fds = FederatedDataset(
    #     dataset="flwrlabs/celeba",
    #     partitioners={"train": NaturalIdPartitioner(partition_by="celeb_id", num_partitions=NUM_CLIENTS)}
    # )
    fds = FederatedDataset(dataset=DATASET, partitioners={"train": NUM_CLIENTS})

    full_dataset = fds.load_split("train")
    full_dataset = full_dataset.cast_column("celeb_id", datasets.Value("int64"))  # ✅ force int
    full_dataset = full_dataset.with_format("numpy")
    full_dataset = full_dataset.filter(lambda ex: ex["celeb_id"] < TOTAL_IDENTITIES_USED)

    ids = set(full_dataset["celeb_id"])
    print(f"[Sanity] Min ID: {min(ids)}, Max ID: {max(ids)}")

    # Group by identity
    id_to_indices = defaultdict(list)
    for idx, identity in enumerate(full_dataset["celeb_id"]):
        id_to_indices[identity].append(idx)


    print(f"[Sanity] id_to_indices has identity 620? {'Yes' if 620 in id_to_indices else 'No'}")

    start = partition_id * IDENTITIES_PER_CLIENT
    end = start + IDENTITIES_PER_CLIENT


    client_ids = list(range(start, end))  # will directly be [500–999] for partition 1

    print(f"[Debug] Partition {partition_id} owns IDs: {client_ids[:3]}...{client_ids[-3:]}")
    print(f"[Debug] Does it include target {target_class}? {'Yes' if target_class in client_ids else 'No'}")

    client_indices = []
    for pid in client_ids:
        if pid in id_to_indices:
            client_indices.extend(id_to_indices[pid])

    print(f"[Debug] Total samples for identity {target_class}: {len(id_to_indices[target_class])}")

    client_data = full_dataset.select(client_indices)
    partition_train_test = client_data.train_test_split(test_size=0.2, seed=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels for RGB
    ])

    def apply_transforms(batch):
        batch["image"] = torch.stack([
            transform(Image.fromarray(img).convert("RGB") if isinstance(img, np.ndarray) else img.convert("RGB"))
            for img in batch["image"]
        ])
        batch["label"] = batch["celeb_id"]
        return batch

    partition_train_test = partition_train_test.map(apply_transforms, batched=True)
    partition_train_test["train"].set_format(type="torch", columns=["image", "label"])
    partition_train_test["test"].set_format(type="torch", columns=["image", "label"])

    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    # Return example target class samples for visualization if requested (e.g., adversary debug)
    if return_target_samples and target_class is not None:
        if target_class in client_ids:
            example_indices = id_to_indices.get(target_class, [])[:16]
            if example_indices:
                example_images = full_dataset.select(example_indices)["image"]
                return trainloader, valloader, target_class, example_images
            else:
                print(f"[Warning] No indices found for target class {target_class}")
                return trainloader, valloader, target_class, []
        else:
            print(f"[Warning] Target class {target_class} not in this client (partition {partition_id})")
            return trainloader, valloader, target_class, []

    return trainloader, valloader, target_class, []


class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=NUM_CLASSES, out_channels=3, feature_maps=128):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = noise_dim + num_classes
        self.project = nn.Sequential(
            nn.Linear(input_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),      # 16x16 -> 32x32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1),          # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_emb(labels)
        input_vec = torch.cat([z, label_embed], dim=1)
        out = self.project(input_vec)
        out = out.view(out.size(0), -1, 4, 4)
        return self.net(out)



class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dynamically determine output size of convs
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 64, 64)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class GANAdversaryClient(NumPyClient):
    def __init__(self, generator, global_model, trainloader, valloader):
        self.generator = generator.to(DEVICE)
        self.global_model = global_model.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.latent_dim = 100
        self.writer = SummaryWriter(log_dir=TB_DIR)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

    def fit(self, parameters, config):
        round_num = config.get("server_round", 0)
        print(f"[Adversary] Round {round_num} started")
        self.set_model_params(parameters)

        if round_num >= START_GAN_TRAINING:
            steps = min(1000 + 500 * (round_num - START_GAN_TRAINING), GAN_EPOCHS)
            self.train_generator_with_global_model_gradient(round_num, steps)
            torch.save(self.generator.state_dict(), os.path.join(GAN_MODELS_DIR, "generator.pth"))

        return self.get_parameters(config), len(self.trainloader), {"server_round": round_num}

    def evaluate(self, parameters, config):
        return 0.0, len(self.valloader), {}

    def set_model_params(self, parameters):
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), parameters)})
        self.global_model.load_state_dict(state_dict, strict=True)

    ...

    def train_generator_with_global_model_gradient(self, round_num, steps=1000):
        self.generator.train()
        self.global_model.eval()

        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        confidence_loss_weight = 1.0
        diversity_loss_weight = 0.1

        print(f"\U0001F3AF [Round {round_num}] Target class for generator inversion: {TARGET_CLASS}")

        for step in range(steps):
            z = torch.randn(BATCH_SIZE, self.latent_dim).to(DEVICE)
            target_labels = torch.full((BATCH_SIZE,), TARGET_CLASS, dtype=torch.long, device=DEVICE)

            fake_imgs = self.generator(z, target_labels)
            logits = self.global_model(fake_imgs)

            confidence_loss = -logits[:, TARGET_CLASS].mean()

            with torch.no_grad():
                features = self.global_model.features(fake_imgs)
                features = features.view(features.size(0), -1)  # Flatten the feature map

            half = BATCH_SIZE // 2
            diversity_loss = -F.l1_loss(features[:half], features[half:])

            loss = confidence_loss_weight * confidence_loss + diversity_loss_weight * diversity_loss

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()

            if (step + 1) % 50 == 0:
                print(
                    f"[Round {round_num} | Step {step}] Inversion Loss: {loss.item():.4f} | "
                    f"Confidence: {confidence_loss.item():.4f} | Diversity: {diversity_loss.item():.4f}")

                self.generator.eval()
                with torch.no_grad():
                    z_vis = torch.randn(16, self.latent_dim).to(DEVICE)
                    print(f"[Debug] z_vis std: {z_vis.std(dim=0).mean().item():.4f} | "
                          f"min: {z_vis.min().item():.2f}, max: {z_vis.max().item():.2f}")
                    print("[Debug] First z_vis sample:", z_vis[0][:5].detach().cpu().numpy())

                    labels_vis = torch.full((16,), TARGET_CLASS, device=DEVICE)
                    fake_imgs_vis = self.generator(z_vis, labels_vis)
                    img_vis = (fake_imgs_vis + 1) / 2
                    grid = torchvision.utils.make_grid(img_vis, nrow=4, normalize=False)

                    self.writer.add_image(f"Generator/Round_{round_num}/Class_{TARGET_CLASS}/Inversion", grid,
                                          global_step=step)
                    torchvision.utils.save_image(grid,
                                                 os.path.join(RUN_DIR,
                                                              f"round{round_num:02}_class{TARGET_CLASS}_step{step}.png"))
                self.generator.train()
                self.writer.add_scalar(f"Generator/Round_{round_num}/Class_{TARGET_CLASS}/Inversion_Loss", loss.item(),
                                       global_step=step)


# Final client function
...

def train(model, dataloader, epochs=1, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for batch in dataloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        if verbose:
            print(f"Epoch {epoch+1}: Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}")

def test(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# Updated client_fn to make only partition 0 adversarial

def client_fn(context: Context) -> Client:
    global generator
    partition_id = context.node_config["partition-id"]

    generator_path = Path(GAN_MODELS_DIR) / "generator.pth"

    trainloader, valloader, class_id, class_images = load_datasets(
        partition_id=partition_id, return_target_samples=True, target_class=TARGET_CLASS)

    print(f'class_images size is: {len(class_images)}')
    global TARGET_DETAILS_LOGGING
    if TARGET_DETAILS_LOGGING and class_images is not None and len(class_images) > 0:
        # Save target reference images
        if class_images is not None and len(class_images) > 0:
            print(f"[Info] Logging reference images for target class {class_id}...")
            tensor_images = torch.stack([
                transforms.ToTensor()(Image.fromarray(img) if isinstance(img, np.ndarray) else img)
                for img in class_images
            ])
            tensor_images = transforms.Normalize((0.5,), (0.5,))(tensor_images)  # Normalize if needed

            grid = torchvision.utils.make_grid(tensor_images, nrow=4, normalize=True)

            # Save to disk
            save_path = os.path.join(RUN_DIR, f"target_identity_{class_id}_examples.png")
            torchvision.utils.save_image(grid, save_path)

            # Log to TensorBoard
            writer = SummaryWriter(log_dir=TB_DIR)
            writer.add_image(f"TargetClass/{class_id}/RealImages", grid, global_step=0)
            writer.close()
        else:
            print(f"[Warning] No target images found for class {class_id}")

        TARGET_DETAILS_LOGGING = False


    if partition_id == 0:
        print("\U0001F310 Adversarial Client Activated")
        generator = Generator()
        if generator_path.exists():
            print("[Client] Loading existing Generator")
            generator.load_state_dict(torch.load(generator_path))
        else:
            print("[Client] Initializing new Generator")

        global_model = Discriminator()
        return GANAdversaryClient(
            generator=generator,
            global_model=global_model,
            trainloader=trainloader,
            valloader=valloader
        ).to_client()

    else:
        print(f"[Client {partition_id}] Standard FlowerClient")
        model = Discriminator().to(DEVICE)
        return FlowerClient(model, trainloader, valloader).to_client()



# Updated FlowerClient class to support standard client training
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_model_params(parameters)
        round_num = config.get("server_round", 0)
        print(f"[Client] Round {round_num}: Training standard model")
        train(self.model, self.trainloader, epochs=CLIENT_EPOCHS, verbose=True)
        return self.get_parameters(config), len(self.trainloader), {"server_round": round_num}

    def evaluate(self, parameters, config):
        self.set_model_params(parameters)
        loss, acc = test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(acc)}

    def set_model_params(self, parameters):
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)


# Run FL

from flwr.server.strategy import FedAvg

class FedAvgWithRound(FedAvg):
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=int(self.fraction_fit * len(client_manager.all())),
            min_num_clients=self.min_fit_clients,
        )
        fit_ins = FitIns(parameters=parameters, config={"server_round": server_round})
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(
            num_clients=int(self.fraction_evaluate * len(client_manager.all())),
            min_num_clients=self.min_evaluate_clients,
        )
        eval_ins = EvaluateIns(parameters=parameters, config={"server_round": server_round})
        return [(client, eval_ins) for client in clients]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute a weighted average of metrics."""
    aggregated = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if not metrics or total_examples == 0:
        return {}

    for key in metrics[0][1].keys():
        total = sum(num_examples * m[key] for num_examples, m in metrics)
        aggregated[key] = total / total_examples

    return aggregated


strategy = FedAvgWithRound(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=(NUM_CLIENTS // 2) + 1,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)


def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)

def main():
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    backend_config = {"client_resources": {"num_cpus": 10, "num_gpus": 0.5 if DEVICE == "cuda" else 0.0}}

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config
    )

if __name__ == "__main__":
    main()

