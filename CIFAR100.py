from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Download CIFAR-100 training and test datasets
full_train_dataset = CIFAR100(root="./data", train=True, transform=transform, download=True)
test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)

# Split training dataset into train and validation sets
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
val_size = len(full_train_dataset) - train_size  # 20% for validation
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print dataset sizes
print(f"Training set size: {len(train_subset)}")
print(f"Validation set size: {len(val_subset)}")
print(f"Test set size: {len(test_dataset)}")



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
