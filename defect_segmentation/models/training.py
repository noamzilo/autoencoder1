import torch
from defect_segmentation.data_loading.DatasetSingleImage import dataset_single_image_default
import numpy as np
from defect_segmentation.models.BasicAutoencoder import BasicAutoencoder
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Utils.ConfigProvider import ConfigProvider
from defect_segmentation.data_loading.DatasetSingleImage import DatasetSingleImage


def train_autoencoder():
    print(f"running with device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    seed = 42
    np.random.seed(seed)
    batch_size = 4
    num_workers = 0

    sample_shape = (10, 10)
    strides = (25, 25)

    is_plotting = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bae = BasicAutoencoder(10 * 10).to(device)

    optimizer = optim.Adam(bae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    path = ConfigProvider.config().data.defective_inspected_path1
    dataset = DatasetSingleImage(path, sample_shape, strides)

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
        for batch_features_train_ in train_loader:
            batch_features_train = batch_features_train_.float().view(-1, np.prod(sample_shape)).to(device)

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
        batch_features_test_ = []
        outputs = []
        for batch_features_test_ in test_loader:
            batch_features_test = batch_features_test_.float().view(-1, np.prod(sample_shape)).to(device)

            outputs = bae(batch_features_test)
            test_loss = criterion(outputs, batch_features_test)

            test_loss.backward()
            test_loss += test_loss.item()

        test_loss = test_loss / len(test_loader)
        test_losses[epoch] = test_loss
        print(f"test : {epoch + 1}/{epochs}, loss = {test_loss:.6f}")


        if is_plotting and epoch % 50 == 0:
            num_samples = batch_features_test_.shape[0]
            fig, axs = plt.subplots(2, num_samples)
            for i_sample in range(num_samples):
                for i in range(2):
                    ax = axs[i, i_sample]
                    if i == 0:
                        ax.imshow(batch_features_test_[i_sample, :, :])
                    else:
                        ax.imshow((outputs.view(4, 10, 10, 3)/255).cpu().detach().numpy()[i_sample, :, :])
                    ax.axis("off")
                    ax.set_title(f"Sample #{i_sample}")
                    plt.pause(0.001)
            plt.close(fig)

    plt.figure()
    plt.plot(np.arange(epochs), train_losses, color='r', label='train loss')
    plt.plot(np.arange(epochs), test_losses, color='b', label='test loss')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    train_autoencoder()
