import sys
from flwr.server.client_proxy import ClientProxy
print(sys.executable)
from collections import OrderedDict

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
from collections import defaultdict
import random
from pathlib import Path



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

DATASET = "mnist"
NUM_CLIENTS = 2
BATCH_SIZE = 64
CLIENT_EPOCHS = 10 # Number of epochs the local models are run by each client node
GAN_EPOCHS = 20000 # Number of Epochs our GAN generator is trained by the Adversary
NUM_CLASSES = 10 # Number of classes for the model. Even though each client has data with only 2 classes, still we will set the class as 10 for each local model to keep the update consistent with the global model
LABELS_PER_CLIENT = 5  # You can change this to any number ≤ total number of classes
NUM_ROUNDS = 5

generator = None
START_GAN_TRAINING = NUM_ROUNDS//2

# At the top level of your script
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join("gan_outputs", timestamp)
os.makedirs(RUN_DIR, exist_ok=True)
TB_DIR = os.path.join(RUN_DIR, "tensorboard")
os.makedirs(TB_DIR, exist_ok=True)
GAN_MODELS_DIR = os.path.join(RUN_DIR, "GAN_Models")
os.makedirs(GAN_MODELS_DIR, exist_ok=True)

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset=DATASET, partitioners={"train": NUM_CLIENTS})
    full_dataset = fds.load_split("train")

    global NUM_CLASSES

    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(full_dataset["label"]):
        label_to_indices[label].append(idx)

    for indices in label_to_indices.values():
        random.shuffle(indices)

    # Assign labels to clients deterministically
    all_labels = sorted(label_to_indices.keys())
    total_label_groups = len(all_labels) // LABELS_PER_CLIENT
    if NUM_CLIENTS > total_label_groups:
        raise ValueError("Too many clients for the number of available label groups.")

    # Each client gets a distinct group of labels
    label_groups = [
        all_labels[i * LABELS_PER_CLIENT : (i + 1) * LABELS_PER_CLIENT]
        for i in range(total_label_groups)
    ]
    client_labels = label_groups[partition_id % len(label_groups)]

    # Gather sample indices for selected labels
    client_indices = []
    for label in client_labels:
        count = len(label_to_indices[label]) // total_label_groups
        client_indices.extend(label_to_indices[label][:count])

    # Subset dataset
    client_data = full_dataset.select(client_indices)

    # Split into train/test
    partition_train_test = client_data.train_test_split(test_size=0.2, seed=42)

    # Transforms
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def apply_transforms(batch):
        if "image" in batch:
            batch["image"] = torch.stack([
                pytorch_transforms(Image.fromarray(img) if isinstance(img, np.ndarray) else img)
                for img in batch["image"]
            ])
        return batch

    partition_train_test = partition_train_test.map(apply_transforms, batched=True)
    partition_train_test["train"].set_format(type="torch", columns=["image", "label"])
    partition_train_test["test"].set_format(type="torch", columns=["image", "label"])

    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)

    testset = fds.load_split("test").map(apply_transforms, batched=True)
    testset.set_format(type="torch", columns=["image", "label"])
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    first_batch = next(iter(trainloader))
    image_sample = first_batch["image"][0]
    input_channels = image_sample.shape[0]
    input_size = (image_sample.shape[1], image_sample.shape[2])
    #num_classes = len(set(int(label) for label in partition_train_test["train"]["label"]))
    #num_classes = NUM_CLASSES
    num_classes = NUM_CLASSES = len(set(full_dataset["label"]))

    return trainloader, valloader, testloader, input_channels, input_size, num_classes


# Dynamic CNN architecture

class Net(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_size: Tuple[int, int] = (28, 28),
        conv_filters: List[int] = [6, 16],
        num_classes: int = 10
    ) -> None:
        super(Net, self).__init__()
        self.convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Build convolutional layers
        in_channels = input_channels
        size = input_size
        for out_channels in conv_filters:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=5))
            size = ((size[0] - 4) // 2, (size[1] - 4) // 2)  # conv then pool
            in_channels = out_channels

        self.flattened_size = in_channels * size[0] * size[1]

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Train and Test loop
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# This gets the gloabl model's params
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# This sends the local model's param
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Flower Client Node
#get_parameters: Return the current local model parameters
#fit: Receive model parameters from the server, train the model on the local data, and return the updated model parameters to the server
#evaluate: Receive model parameters from the server, evaluate the model on the local data, and return the evaluation result to the server

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        try:
            set_parameters(self.net, parameters)
            round_num = config.get("server_round", 0)
            print(f'server round received in fit method: {round_num}')
            train(self.net, self.trainloader, epochs=CLIENT_EPOCHS, verbose=True)
            return get_parameters(self.net), len(self.trainloader), {"server_round": round_num}
        except Exception as e:
            print(f"[Client {os.getpid()}] Fit failed: {e}")
            raise e
    def evaluate(self, parameters, config):
        print(f'params inside client evaluate: {parameters}')
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}




def client_fn(context: Context) -> Client:
    global generator
    try:
        partition_id = context.node_config["partition-id"]
        trainloader, valloader, _, input_channels, input_size, num_classes = load_datasets(partition_id)

        generator_path = Path(os.path.join(GAN_MODELS_DIR, "global_generator.pth"))

        if partition_id == 0:
            print("🌐 Adversarial Client Activated")
            generator = Generator(noise_dim=100, out_channels=1)
            cnn_model = Net(input_channels, input_size, [6, 16], num_classes).to(DEVICE)
            if generator_path.exists():
                print("Loading existing Generator")
                generator.load_state_dict(torch.load(generator_path))
            else:
                print("Initializing new Generator")
            return GANAdversaryClient(
                global_cnn=cnn_model,
                trainloader=trainloader,
                generator=generator,
                latent_dim=100,
                valloader=valloader,
                run_dir=RUN_DIR,
                tb_dir=TB_DIR,
            ).to_client()

        else:
            net = Net(input_channels, input_size, [6, 16], num_classes).to(DEVICE)
            return FlowerClient(net, trainloader, valloader).to_client()
    except Exception as e:
        print(f"[Client {context.node_config.get('partition-id', '?')}] Failed in client_fn: {e}")
        raise


# Server Side averaging

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if not metrics or total_examples == 0:
        return {}

    metric_keys = metrics[0][1].keys()
    for key in metric_keys:
        total = sum(num_examples * m[key] for num_examples, m in metrics)
        aggregated[key] = total / total_examples

    return aggregated




class FedAvgWithRound(FedAvg):
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training by passing `server_round`."""
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
        """Configure evaluation and pass `server_round` to clients."""
        clients = client_manager.sample(
            num_clients=int(self.fraction_evaluate * len(client_manager.all())),
            min_num_clients=self.min_evaluate_clients,
        )
        eval_ins = EvaluateIns(parameters=parameters, config={"server_round": server_round})
        return [(client, eval_ins) for client in clients]



# Create FedAvg strategy
strategy = FedAvgWithRound(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=(NUM_CLIENTS // 2) + 1,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
)

#Generator to be used by the Adversary
class Generator(nn.Module):
    def __init__(self, noise_dim=100, out_channels=1, feature_maps=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_maps * 4, 3, 1, 0, bias=False),  # 1x1 -> 3x3
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),  # 3x3 -> 7x7
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),  # 7x7 -> 14x14
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1, bias=False),  # 14x14 -> 28x28
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class GANAdversaryClient(NumPyClient):

    def __init__(self, global_cnn, trainloader, generator, latent_dim, valloader, run_dir, tb_dir):
        self.global_cnn = global_cnn.to(DEVICE)
        self.trainloader = trainloader
        self.generator = generator.to(DEVICE)
        self.latent_dim = latent_dim
        self.valloader = valloader
        self.run_dir = run_dir
        self.tb_dir = tb_dir


        # Save run configuration
        run_config = {
            "timestamp": timestamp,
            "batch_size": BATCH_SIZE,
            "latent_dim": latent_dim,
            "generator_lr": 2e-4,
            "dummy_model_lr": 1e-3,
            "device": DEVICE,
            "num_clients": NUM_CLIENTS,
            "labels_per_client": LABELS_PER_CLIENT,
            "CLIENT_EPOCHS": CLIENT_EPOCHS,
            "GAN_EPOCHS": GAN_EPOCHS,
            "gan_generator_steps": 10,
            "description": "GAN adversary FL run with synthetic data and dummy model update"
        }

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=4)

        self.writer = SummaryWriter(log_dir=self.tb_dir)

        #self.dummy_model = Net(input_channels=1, input_size=(28, 28), conv_filters=[6, 16], num_classes=10).to(DEVICE)
        #self.dummy_optimizer = torch.optim.Adam(self.dummy_model.parameters(), lr=1e-3)
        #self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.global_cnn)

    def fit(self, parameters, config):
        # 1. Update dummy model with global weights
        set_parameters(self.global_cnn, parameters)
        round_num = config.get("server_round", 0)
        print(f'server round received in fit method: {round_num}')
        train(self.global_cnn, self.trainloader, epochs=CLIENT_EPOCHS, verbose=True)

        # 2. Train dummy model on random data with fake labels (deceptive update)
        # self.dummy_model.train()
        # for _ in range(1):  # minimal 1 epoch
        #     images = torch.randn(BATCH_SIZE, 1, 28, 28).to(DEVICE)
        #     labels = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
        #     self.dummy_optimizer.zero_grad()
        #     outputs = self.dummy_model(images)
        #     loss = self.criterion(outputs, labels)
        #     loss.backward()
        #     self.dummy_optimizer.step()

        # 3. Clone global model as Discriminator (frozen)
        discriminator = Net(
            input_channels=1, input_size=(28, 28), conv_filters=[6, 16], num_classes=NUM_CLASSES
        ).to(DEVICE)
        set_parameters(discriminator, parameters)
        discriminator.eval()

        # 4. Train Generator to fool CNN (GAN step)
        if round_num >= NUM_ROUNDS//2:
            steps = min(500 + 500 * (round_num - START_GAN_TRAINING), 200000)  # Progressive GAN training
            self.train_generator(discriminator, round_num, steps=steps)
            generator_path = Path(os.path.join(GAN_MODELS_DIR, "global_generator.pth"))
            torch.save(self.generator.state_dict(), generator_path)

        # Return deceptive updates to appear honest
        #return get_parameters(self.dummy_model), BATCH_SIZE, {"server_round": round_num}
        return get_parameters(self.global_cnn), len(self.trainloader), {"server_round": round_num}

    def train_generator(self, discriminator, round_num, steps=10):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        criterion = nn.CrossEntropyLoss()
        target_class = 8

        for step in range(steps):
            z = torch.randn(BATCH_SIZE, self.latent_dim, 1, 1).to(DEVICE)
            fake_imgs = self.generator(z)

            # Use the frozen CNN as a Discriminator
            # outputs = discriminator(fake_imgs)
            # if isinstance(outputs, torch.Tensor) and outputs.ndim == 2:
            #     outputs = outputs.softmax(dim=1)
            # Use confidence on target digit (or total entropy) as the feedback signal

            logits = discriminator(fake_imgs)
            #labels = torch.randint(0, NUM_CLASSES, (logits.shape[0],), device=DEVICE)
            # Ensure labels match the actual batch size
            labels = torch.full((logits.shape[0],), target_class, dtype=torch.long, device=DEVICE)
            loss = criterion(logits, labels)

            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()
            # Print loss per step
            print(f"[Round {round_num} | Step {step + 1}/{steps}] Generator Loss: {loss.item():.4f}")

            if step % 50 == 0:
                self.generator.eval()
                with torch.no_grad():
                    z_vis = torch.randn(16, self.latent_dim, 1, 1).to(DEVICE)
                    fake_imgs_vis = self.generator(z_vis)
                    img_grid = torchvision.utils.make_grid(fake_imgs_vis, nrow=4, normalize=True)
                    self.writer.add_image("Generator/FakeImages", img_grid, global_step=step)
                    torchvision.utils.save_image(img_grid, os.path.join(self.run_dir, f"round_{round_num:03}_{step}.png"))

                self.generator.train()
                # Log scalar loss
                self.writer.add_scalar("Generator/Loss", loss.item(), global_step=step)


    def evaluate(self, parameters, config):
        # set_parameters(self.dummy_model, parameters)
        # loss, acc = test(self.dummy_model, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(acc)}

        print(f'params inside client evaluate: {parameters}')
        set_parameters(self.global_cnn, parameters)
        loss, accuracy = test(self.global_cnn, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)


def main():
    #Setup Federated Learning

    trainloader, _, _, input_channels, input_size, num_classes = load_datasets(partition_id=0)
    print(f"Data shape: {input_channels}x{input_size}, Num classes: {num_classes}")

    batch = next(iter(trainloader))
    images, labels = batch["image"], batch["label"]

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.squeeze(1).numpy()

    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()

    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if DEVICE == "cuda":
        backend_config = {"client_resources": {"num_cpus": 20, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

def debugger():
    dummy_context = Context(
        run_id="test-run",
        node_id="test-node",
        state=None,
        run_config={},
        node_config={"partition-id": 0}
    )
    c = client_fn(dummy_context)
    print("Client created successfully")

if __name__ == "__main__":
    main()
    #test()
