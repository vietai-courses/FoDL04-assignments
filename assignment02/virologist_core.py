# virologist_core.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    data_loader: DataLoader, 
    device: torch.device, 
    epoch_index: int, 
    log_interval: int = 100
) -> None:
    """Executes one full training epoch over the provided dataset."""
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            dataset_size = len(data_loader.dataset)
            current_progress = batch_idx * len(data)
            percentage = 100. * batch_idx / len(data_loader)
            
            print(f"Viral Injection [Epoch {epoch_index}]: "
                  f"[{current_progress:>5}/{dataset_size} "
                  f"({percentage:>3.0f}%)]\t"
                  f"Loss: {loss.item():.6f}")

def test_model(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
) -> float:
    """Evaluates the model's accuracy on a given dataset."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    accuracy = correct / len(data_loader.dataset)
    return accuracy

class VirusValidator:
    """Auto-Grader logic to verify student network architectures."""
    
    @staticmethod
    def verify_alpha_strain(epochs_taken: int, accuracy: float) -> None:
        if epochs_taken <= 5 and accuracy >= 0.98:
            print(f"\n[SUCCESS] Alpha Strain survived! (Epochs: {epochs_taken}, Acc: {accuracy:.2%})")
        else:
            print(f"\n[FAIL] Virus neutralized. (Epochs: {epochs_taken}/5, Acc: {accuracy:.2%}/98.0%)")

    @staticmethod
    def get_parameter_count(model: nn.Module) -> int:
        return sum(tensor.numel() for tensor in model.parameters() if tensor.requires_grad)

    @staticmethod
    def verify_beta_strain(model, train_func, test_func, optimizer, loader, device) -> None:
        total_params = VirusValidator.get_parameter_count(model)
        if total_params > 2000:
            print(f"\n[FAIL] Payload too large: {total_params} parameters (Limit: 2000).")
            return

        print(f"\n[PASSED] Size Check: {total_params} parameters. Initiating training...")
        for epoch in range(1, 6):
            train_func(model, optimizer, loader, device, epoch)
            current_accuracy = test_func(model, loader, device)
            if current_accuracy > 0.90:
                print(f"\n[SUCCESS] Beta Virus survived with accuracy {current_accuracy:.2%}!")
                return
        print(f"\n[FAIL] Virus too weak. Final Accuracy: {current_accuracy:.2%} (Target: 90%)")

    @staticmethod
    def verify_gamma_strain(deep_predictions: torch.Tensor, collapsed_predictions: torch.Tensor) -> None:
        maximum_difference = torch.max(torch.abs(deep_predictions - collapsed_predictions)).item()
        if maximum_difference < 1e-4:
            print(f"\n[SUCCESS] Linear collapse verified. (Max Diff: {maximum_difference:.6f})")
        else:
            print(f"\n[FAIL] Collapse incorrect. (Max Diff: {maximum_difference:.6f})")

    @staticmethod
    def verify_omicron_strain(model_instance: nn.Module) -> None:
        for layer_name, layer in model_instance.named_modules():
            if 'Pool' in type(layer).__name__:
                print(f"\n[FAIL] Illegal architecture. '{layer_name}' utilizes blocked Pooling.")
                return
                
        test_tensor = torch.randn(1, 1, 28, 28)
        actual_shape = model_instance(test_tensor).shape
        target_shape = (1, 10, 4, 4)
        
        if actual_shape == target_shape:
            print(f"\n[SUCCESS] Omicron Geometry Verified: {actual_shape}")
        else:
            print(f"\n[FAIL] Target geometry missed. Expected {target_shape}, got {actual_shape}")