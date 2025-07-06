import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hydra.utils import instantiate


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    print("running training with config: ", cfg)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    trsmfs = transforms.Compose([transforms.ToTensor()])
    train_dataset = instantiate(cfg.dataset, transform=trsmfs)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {total_loss:.4f}")


if __name__ == "__main__":
    main()
