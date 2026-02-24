import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

MAX_EPOCHS = 5
ALPHA_REQUIRED_ACC = 0.98
BETA_TARGET_ACC = 0.80
PARAM_LIMIT = 2000
GAMMA_TOLERANCE = 1e-4
OMICRON_DEFAULT_INPUT_SHAPE: Tuple[int, ...] = (1, 1, 28, 28)
OMICRON_DEFAULT_TARGET_SHAPE: Tuple[int, ...] = (1, 10, 4, 4)

def train_one_epoch(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    data_loader: DataLoader, 
    device: torch.device, 
    epoch_index: int, 
    log_interval: int = 100
) -> None:
    """
    Executes one full training epoch over the provided dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch neural network model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer algorithm (e.g., Adam, SGD) updating the model weights.
    data_loader : torch.utils.data.DataLoader
        The DataLoader providing the training batches.
    device : torch.device
        The hardware device ('cpu' or 'cuda') to execute the training on.
    epoch_index : int
        The current epoch number (used for logging progress).
    log_interval : int, optional
        The number of batches to process before printing a progress log. Default is 100.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Controlled Progress Printing
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
    """
    Evaluates the model's accuracy on a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch neural network model to evaluate.
    data_loader : torch.utils.data.DataLoader
        The DataLoader providing the testing/validation batches.
    device : torch.device
        The hardware device ('cpu' or 'cuda') to execute the evaluation on.

    Returns
    -------
    float
        The overall accuracy of the model on the provided dataset (between 0.0 and 1.0).
    """
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    accuracy = correct / len(data_loader.dataset)
    return accuracy


class VirusValidator:
    """Collection of static validation helpers for model 'strains'.

    Each `verify_*` method performs a specific check used by the assignment
    and logs a concise result. The methods return `True` on success and
    `False` on failure to allow programmatic checks in addition to logging.
    """
    @staticmethod
    def verify_alpha_strain(epochs_taken: int, accuracy: float) -> bool:
        """Check the Alpha strain: fast convergence and high accuracy.

        Parameters
        ----------
        epochs_taken : int
            Number of epochs used during training.
        accuracy : float
            Final accuracy value (0.0 - 1.0).

        Returns
        -------
        bool
            True if both epoch and accuracy thresholds are met, else False.
        """
        is_fast_enough = epochs_taken <= MAX_EPOCHS
        is_accurate_enough = accuracy >= ALPHA_REQUIRED_ACC

        if is_fast_enough and is_accurate_enough:
            logging.info("[SUCCESS] Alpha Strain survived! (Epochs: %d, Acc: %.2f%%)", epochs_taken, accuracy * 100)
            return True
        else:
            logging.warning("[FAIL] Virus neutralized. (Epochs: %d/%d, Acc: %.2f%%/%.2f%%)", epochs_taken, MAX_EPOCHS, accuracy * 100, ALPHA_REQUIRED_ACC * 100)
            return False

    @staticmethod
    def get_parameter_count(model: nn.Module) -> int:
        return sum(tensor.numel() for tensor in model.parameters() if tensor.requires_grad)

    @staticmethod
    def verify_beta_strain(
        model: nn.Module,
        train_func: Callable[[nn.Module, optim.Optimizer, DataLoader, torch.device, int], None],
        test_func: Callable[[nn.Module, DataLoader, torch.device], float],
        optimizer: optim.Optimizer,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> bool:
        """Check Beta strain: parameter size and short training to target accuracy.

        The function now requires `train_loader`, `test_loader`, and `device`
        to call the provided `train_func` and `test_func` which operate on
        batches and devices. Returns True if the model passes the
        parameter-size check and achieves `BETA_TARGET_ACC` within
        `MAX_EPOCHS`, otherwise False.
        """
        total_params = VirusValidator.get_parameter_count(model)

        if total_params > PARAM_LIMIT:
            logging.error("[FAIL] Payload too large: %d parameters (Limit: %d).", total_params, PARAM_LIMIT)
            return False

        logging.info("[PASSED] Size Check: %d parameters. Initiating training...", total_params)

        if train_loader is None or test_loader is None or device is None:
            logging.error("Missing `train_loader`, `test_loader`, or `device` for training.")
            return False

        current_accuracy = 0.0
        for epoch in range(1, MAX_EPOCHS + 1):
            train_func(model, optimizer, train_loader, device, epoch)
            current_accuracy = test_func(model, test_loader, device)

            if current_accuracy > BETA_TARGET_ACC:
                logging.info("[SUCCESS] Beta Virus survived with accuracy %.2f%%!", current_accuracy * 100)
                return True

        logging.warning("[FAIL] Virus too weak. Final Accuracy: %.2f%% (Target: %.2f%%)", current_accuracy * 100, BETA_TARGET_ACC * 100)
        return False

    @staticmethod
    def verify_gamma_strain(deep_predictions: torch.Tensor, collapsed_predictions: torch.Tensor) -> bool:
        """Compare two prediction tensors and assert they're nearly identical.

        Returns True if the maximum absolute difference is below `GAMMA_TOLERANCE`.
        """
        maximum_difference = torch.max(torch.abs(deep_predictions - collapsed_predictions)).item()
        is_mathematically_identical = maximum_difference < GAMMA_TOLERANCE

        if is_mathematically_identical:
            logging.info("[SUCCESS] Linear collapse verified. (Max Diff: %.6f)", maximum_difference)
            return True
        else:
            logging.error("[FAIL] Collapse incorrect. (Max Diff: %.6f)", maximum_difference)
            return False

    @staticmethod
    def verify_omicron_strain(
        model_instance: nn.Module,
        test_tensor: Optional[torch.Tensor] = None,
        target_shape: Tuple[int, ...] = OMICRON_DEFAULT_TARGET_SHAPE,
    ) -> bool:
        """Verify the model architecture and output geometry for Omicron.

        Parameters
        ----------
        model_instance : nn.Module
            The model to inspect and run with a test input.
        test_tensor : Optional[torch.Tensor]
            Optional input tensor to feed the model. If None, a random tensor
            with `OMICRON_DEFAULT_INPUT_SHAPE` will be used.
        target_shape : Tuple[int, ...]
            Expected output shape produced by the model.

        Returns
        -------
        bool
            True if the architecture and geometry checks pass, else False.
        """
        # 1. Inspect Architecture for illegal layers
        for layer_name, layer in model_instance.named_modules():
            if 'Pool' in type(layer).__name__:
                logging.error("[FAIL] Illegal architecture. '%s' utilizes blocked Pooling.", layer_name)
                return False

        # 2. Inspect Geometric Shape
        if test_tensor is None:
            test_tensor = torch.randn(*OMICRON_DEFAULT_INPUT_SHAPE)

        try:
            actual_shape = model_instance(test_tensor).shape
        except Exception as exc:
            logging.exception("[FAIL] Model failed to run on test tensor: %s", exc)
            return False

        if actual_shape == target_shape:
            logging.info("[SUCCESS] Omicron Geometry Verified: %s", actual_shape)
            return True
        else:
            logging.error("[FAIL] Target geometry missed. Expected %s, got %s", target_shape, actual_shape)
            return False