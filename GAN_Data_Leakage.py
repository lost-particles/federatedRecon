import sys
print(sys.executable)
from collections import OrderedDict
from typing import List, Tuple

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
import os
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

dataset = "mnist"


NUM_CLIENTS = 10
BATCH_SIZE = 32

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset=dataset, partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    # Dynamically extract info
    features = fds.dataset_info().features["img"]
    input_channels = features.shape[0]
    input_size = (features.shape[1], features.shape[2])
    num_classes = fds.dataset_info().features["label"].num_classes

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
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
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
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
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
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}




def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _, input_channels, input_size, num_classes = load_datasets(partition_id)

    if partition_id == 0:
        print("🌐 Adversarial Client Activated")
        cnn_model = Net(input_channels, input_size, [6, 16], num_classes)
        generator = Generator(noise_dim=100, out_channels=1)
        return GANAdversaryClient(
            global_cnn=cnn_model,
            generator=generator,
            latent_dim=100,
            valloader=valloader,
        ).to_client()

    else:
        net = Net(input_channels, input_size, [6, 16], num_classes).to(DEVICE)
        return FlowerClient(net, trainloader, valloader).to_client()


# Server Side averaging

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
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
    def __init__(self, global_cnn, generator, latent_dim, valloader):
        self.global_cnn = global_cnn.to(DEVICE)
        self.generator = generator.to(DEVICE)
        self.latent_dim = latent_dim
        self.valloader = valloader
        self.output_dir = "gan_outputs"
        self.tb_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # Dummy model to send updates (trained on random noise)
        self.dummy_model = Net(
            input_channels=1, input_size=(28, 28), conv_filters=[6, 16], num_classes=10
        ).to(DEVICE)

        self.dummy_optimizer = torch.optim.Adam(self.dummy_model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.dummy_model)

    def fit(self, parameters, config):
        # 1. Update dummy model with global weights
        set_parameters(self.dummy_model, parameters)

        # 2. Train dummy model on random data with fake labels (deceptive update)
        self.dummy_model.train()
        for _ in range(1):  # minimal 1 epoch
            images = torch.randn(BATCH_SIZE, 1, 28, 28).to(DEVICE)
            labels = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
            self.dummy_optimizer.zero_grad()
            outputs = self.dummy_model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.dummy_optimizer.step()

        # 3. Clone global model as Discriminator (frozen)
        discriminator = Net(
            input_channels=1, input_size=(28, 28), conv_filters=[6, 16], num_classes=10
        ).to(DEVICE)
        set_parameters(discriminator, parameters)
        discriminator.eval()

        # 4. Train Generator to fool CNN (GAN step)
        self.train_generator(discriminator, config.get("server_round", 0))

        # Return deceptive updates to appear honest
        return get_parameters(self.dummy_model), BATCH_SIZE, {}

    def train_generator(self, discriminator, round_num, steps=10):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)

        for _ in range(steps):
            z = torch.randn(BATCH_SIZE, self.latent_dim, 1, 1).to(DEVICE)
            fake_imgs = self.generator(z)

            # Use the frozen CNN as a Discriminator
            outputs = discriminator(fake_imgs)
            if isinstance(outputs, torch.Tensor) and outputs.ndim == 2:
                outputs = outputs.softmax(dim=1)

            # Use confidence on target digit (or total entropy) as the feedback signal
            target_labels = torch.ones(BATCH_SIZE).to(DEVICE)
            loss = -torch.mean(torch.sum(outputs, dim=1))  # inverse of confidence
            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()

        # Log scalar loss
        self.writer.add_scalar("Generator/Loss", loss.item(), global_step=round_num)

        # Save image grid for TensorBoard
        img_grid = torchvision.utils.make_grid(fake_imgs[:16], nrow=4, normalize=True)
        self.writer.add_image("Generator/FakeImages", img_grid, global_step=round_num)

        # Also save to disk
        torchvision.utils.save_image(img_grid, f"gan_outputs/round_{round_num:03}.png")

    def evaluate(self, parameters, config):
        set_parameters(self.dummy_model, parameters)
        loss, acc = test(self.dummy_model, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(acc)}



def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)


def main():
    #Setup Federated Learning

    trainloader, _, _, input_channels, input_size, num_classes = load_datasets(partition_id=0)
    print(f"Data shape: {input_channels}x{input_size}, Num classes: {num_classes}")

    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]

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
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
