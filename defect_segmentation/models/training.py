import torch
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default
import numpy as np
from defect_segmentation.models.BasicAutoencoder import BasicAutoencoder
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


def train_autoencoder():
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

    epochs = 1000
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            batch_features = batch_features.float().view(-1, 100).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = bae(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


if __name__ == "__main__":
    train_autoencoder()
