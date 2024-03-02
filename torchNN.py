# import dependencies
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# get data
train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=32)
# 1, 28, 28 - classes 0-9

# Image classifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()  # Correctly pass `self` to `super()`
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Use kernel_size and padding
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten after last Conv2d
            nn.Linear(64 * 28 * 28, 10, True)  # Calculate input size after convolutions
        )

    def forward(self, x):
        # print(x.shape)
        return self.model(x)


# instance of our nn, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training Flow
if __name__ == "__main__":
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            yhat = clf(data)
            loss = loss_fn(yhat, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{9}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)
