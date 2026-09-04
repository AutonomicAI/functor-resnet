import numpy as np
import torch
import torch.nn.functional as F


def image_to_vector(image_tensor: torch.Tensor) -> np.ndarray:
    """Canonical image encoding for hashing/anchoring: CHW float32 numpy array."""
    return np.ascontiguousarray(image_tensor.detach().cpu().numpy().astype(np.float32))


class ResNetBaseFunction:
    """
    Wraps a frozen ResNet classifier as a Vector -> Vector base function.

    Output is softmax probabilities (bounded [0, 1], sums to 1) so one-shot
    correction targets are well-defined (a one-hot "predict this class with full
    confidence" vector) and invariant magnitude bounds are meaningful.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
        return probs.squeeze(0).numpy().astype(np.float64)


def one_hot(class_index: int, num_classes: int) -> np.ndarray:
    v = np.zeros(num_classes, dtype=np.float64)
    v[class_index] = 1.0
    return v
