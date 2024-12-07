from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Define transformations (e.g., resizing, normalizing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Download CIFAR-10 training and test datasets
train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

# Split training dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoaders for batch processing
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Summary
print(f"Train samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples: {len(test_dataset)}")



vis = True
iter = 0
for inputs, labels in train_loader:


    print('iter {} of {}'.format(iter, len(train_loader)))
    iter += 1

    # visualize the input and labels
    if vis:
        # mind that the input is normalized to [-1, 1] so we need to denormalize it to [0, 1] to visualize it
        # mind that the input is normalized to [-1, 1] so we need to denormalize it to [0, 1] to visualize it
        inputs = inputs * 0.5 + 0.5
        print('inputs.shape', inputs.shape)
        print('labels.shape', labels.shape)
        # print('inputs', inputs)
        print('labels', labels)
        print('inputs[0].shape', inputs[0].shape)
        # print(inputs[0])
        print('inputs[0].permute(1, 2, 0).shape', inputs[0].permute(1, 2, 0).shape)
        # print(inputs[0].permute(1, 2, 0))
        plt.imshow(inputs[0].permute(1, 2, 0))
        plt.show()
