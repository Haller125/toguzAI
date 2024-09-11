import numpy as np
from torch import nn
import torch
import torch.optim as optim

import torch.ao.quantization as quantization
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Scaling factors (будут меняться)
s_W = 127  # weight scaling factor
s_A = 127  # input scaling factor
s_O = 127  # output scaling factor
s_B = s_W * s_O  # bias scaling factor
s_WA = (s_W * s_O) / s_A  # weight scaling factor for the input weights

# The number of features in the input layer
NUM_FEATURES = 27
M = 4 # Будут меняться
N = 8 # Будут меняться
K = 1

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm):
        white_features = white_features.unsqueeze(0) if white_features.dim() == 1 else white_features
        black_features = black_features.unsqueeze(0) if black_features.dim() == 1 else black_features

        w = self.ft(white_features) # white's perspective
        b = self.ft(black_features) # black's perspective

        # Ensure w and b have at least two dimensions
        if len(w.shape) == 1:
            w = w.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        # Remember that we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm[:, None] * torch.cat([w, b], dim=1)) + ((1 - stm[:, None]) * torch.cat([b, w], dim=1))

        # Apply ClippedReLU (clamp_) and proceed through layers
        l1_x = torch.clamp(accumulator, 0, 127)
        l2_x = torch.clamp(self.l1(l1_x), 0, 127)

        final_output = self.l2(l2_x)

        return final_output

# TODO debug this (console errors)
def train_nnue(nnmodel, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(nnmodel.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        nnmodel.train()
        train_loss = 0
        for white_features, black_features, stm, eval in train_loader:
            optimizer.zero_grad()
            outputs = nnmodel(white_features, black_features, stm)
            outputs = outputs.squeeze(-1)

            if outputs.shape != eval.shape:
                raise ValueError(f"Shape mismatch: outputs {outputs.shape}, eval {eval.shape}")

            loss = mse_loss(outputs, eval)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(nnmodel.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        nnmodel.eval()
        val_loss = 0
        with torch.no_grad():
            for white_features, black_features, stm, eval in val_loader:
                outputs = nnmodel(white_features, black_features, stm)
                outputs = outputs.squeeze(-1)

                if outputs.shape != eval.shape:
                    raise ValueError(f"Shape mismatch: outputs {outputs.shape}, eval {eval.shape}")

                val_loss += mse_loss(outputs, eval).item()

        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        scheduler.step(val_loss)

    return nnmodel

def compute_loss(outputs, targets, epsilon=1e-8):
    loss_eval = (targets * torch.log(targets + epsilon) + (1 - targets) * torch.log(1 - targets + epsilon))
    loss_eval -= (targets * torch.log(outputs + epsilon) + (1 - targets) * torch.log(1 - outputs + epsilon))
    return loss_eval.mean()


def chess_eval_loss(outputs, targets, delta=10.0):
    """
    Custom loss function for chess evaluation:
    - Uses Huber Loss for robustness against outliers
    - Adds a scaling factor to give more weight to small differences around 0
    """
    diff = outputs - targets
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta))
    linear = abs_diff - quadratic
    base_loss = 0.5 * quadratic ** 2 + delta * linear

    # Add scaling factor to emphasize small differences around 0
    scale = 1.0 / (1.0 + torch.abs(targets))

    return (base_loss * scale).mean()


if __name__ == "__main__":
    model = NNUE()
    white_features = torch.rand(10, NUM_FEATURES)
    black_features = torch.rand(10, NUM_FEATURES)
    stm = torch.randint(0, 2, (10, 1))  # side to move, randomly 0 or 1

    # Function to check if tensor contains only integers
    def check_quantization(output):
        if output.dtype in [torch.int8, torch.int16, torch.int32]:
            print(f"Output dtype is {output.dtype}, which is correctly quantized.")
        else:
            print(f"Error: Output dtype is {output.dtype}, expected integer type.")
        if output.is_floating_point():
            print("Error: Output contains floating point numbers.")
        else:
            print("Output contains only integers, quantization is effective.")

    # Running the model and testing outputs
    output = model.forward(white_features, black_features, stm)
    check_quantization(output)