import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt





# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Download CIFAR-10 training and test datasets
full_train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

# Split training dataset into train and validation sets
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
val_size = len(full_train_dataset) - train_size  # 20% for validation
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])


# Load teacher (ViT) and student (ResNet) models. Vision Transformer (ViT-B/16): Parameters: ~86 million (M), ResNet-50: Parameters: ~25.6 million (M). ViT-B/16 has ~3.4x more parameters than ResNet-50.
teacher_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
student_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Freeze teacher model
for param in teacher_model.parameters():
    param.requires_grad = False


# Modify student model for CIFAR-10 (10 classes) instead of ImageNet (1000 classes) to match the dataset.
## CAUTION: This line is commented out because it is semantically incorrect to distill a model trained on ImageNet to a model trained on CIFAR-10/100.
### this also would cause an error in KLDivLoss. The error occurs because the teacher model (vit_b_16) outputs logits for 1,000 classes (from ImageNet), while the student model (resnet50) is modified to output logits for 10 classes (CIFAR-10). The sizes of the tensors for KL divergence don't match.
# student_model.fc = nn.Linear(student_model.fc.in_features, 10)


print('NOTE: This implementation is for convenience and demonstration purposes only unless you use the same dataset and classes for both models. '
      'It is semantically incorrect to distill a model trained on ImageNet to a model trained on CIFAR-10/100. '
      'Because the teacher model is trained on a different dataset (ImageNet) and classes than the student model (CIFAR-10/100). '
      'To solve this and properly distill knowledge from a teacher model to a student model, both models should be trained on the same dataset and classes. '
      'This is not a limitation of the distillation technique itself, but rather a limitation of the current implementation as I didnt have the time to download and train the models on ImageNet due to its large size. ')

# Define distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        distill_loss = self.kl_div(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        return self.alpha * distill_loss + (1 - self.alpha) * ce_loss

# Initialize distillation components
distill_loss_fn = DistillationLoss(alpha=0.5, temperature=4)
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# Use the prepared DataLoaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

num_epochs = 10  # Example: 10 epochs
vis = False




if __name__ == '__main__':
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0

        iter = 0
        for inputs, labels in train_loader:

            iter += 1
            print('iter {} of {}'.format(iter, len(train_loader)))

            # visualize the input and labels
            if vis:
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


            inputs, labels = inputs.to(device), labels.to(device)

            # Teacher model inference
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # Student model inference
            student_logits = student_model(inputs)

            # Compute distillation loss
            loss = distill_loss_fn(student_logits, teacher_logits, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # Validation loop (optional)
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                teacher_logits = teacher_model(inputs)
                student_logits = student_model(inputs)
                loss = distill_loss_fn(student_logits, teacher_logits, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader)}")

    # Evaluate on the test set
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
