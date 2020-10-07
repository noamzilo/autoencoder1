import torch
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default
import numpy as np
from defect_segmentation.models.BasicAutoencoder import BasicAutoencoder
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train_autoencoder():
    print(f"running with device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    seed = 42
    np.random.seed(seed)
    batch_size = 4
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bae = BasicAutoencoder(10 * 10).to(device)

    optimizer = optim.Adam(bae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = dataset_single_image_default()
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = \
        torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, )

    epochs = 300
    train_losses = np.zeros((epochs,))
    test_losses = np.zeros((epochs,))
    for epoch in range(epochs):
        bae.train()
        train_loss = 0
        for batch_features_train in train_loader:
            batch_features_train = batch_features_train.float().view(-1, 100).to(device)

            optimizer.zero_grad()

            outputs = bae(batch_features_train)
            train_loss = criterion(outputs, batch_features_train)

            train_loss.backward()
            optimizer.step()
            train_loss += train_loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses[epoch] = train_loss
        print(f"train : {epoch + 1}/{epochs}, loss = {train_loss:.6f}")

        bae.eval()
        test_loss = 0
        for batch_features_test in test_loader:
            batch_features_test = batch_features_test.float().view(-1, 100).to(device)

            outputs = bae(batch_features_test)
            test_loss = criterion(outputs, batch_features_test)

            test_loss.backward()
            test_loss += test_loss.item()

        test_loss = test_loss / len(test_loader)
        test_losses[epoch] = test_loss
        print(f"test : {epoch + 1}/{epochs}, loss = {test_loss:.6f}")

    plt.figure()
    plt.plot(np.arange(epochs), train_losses, color='r', label='train loss')
    plt.plot(np.arange(epochs), test_losses, color='b', label='test loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    train_autoencoder()
