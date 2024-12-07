import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights

# Load models with updated weights argument
teacher_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
student_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)



# Freeze teacher model
for param in teacher_model.parameters():
    param.requires_grad = False

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

# Dummy data loader
data_loader = [(torch.randn(8, 3, 224, 224), torch.randint(0, 1000, (8,)))]

# Training loop
student_model.train()
for epoch in range(5):  # Short example with 5 epochs
    for inputs, labels in data_loader:
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        student_logits = student_model(inputs)

        loss = distill_loss_fn(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
