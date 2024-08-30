import numpy as np
from torch import nn
import torch
import torch.optim as optim

import torch.ao.quantization as quantization
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
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # Apply ClippedReLU (clamp_) and proceed through layers
        l1_x = torch.clamp(accumulator, 0, 127)
        l2_x = torch.clamp(self.l1(l1_x), 0, 127)

        final_output = self.l2(l2_x)

        return final_output


def train_nnue(nnmodel, train_loader, val_loader, num_epochs=10, learning_rate=0.01):
    criterion = nn.MSELoss()  # or whatever loss function is appropriate
    optimizer = optim.Adam(nnmodel.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        nnmodel.train()
        for white_features, black_features, stm, targets in train_loader:
            optimizer.zero_grad()
            outputs = nnmodel(white_features, black_features, stm)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        nnmodel.eval()
        val_loss = 0
        with torch.no_grad():
            for white_features, black_features, stm, targets in val_loader:
                outputs = nnmodel(white_features, black_features, stm)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")


def compute_loss(wdl_eval_target, wdl_eval_model, game_result, lambda_=0.5, epsilon=1e-8):
    loss_eval = (wdl_eval_target * np.log(wdl_eval_target + epsilon) + (1 - wdl_eval_target) * np.log(
        1 - wdl_eval_target + epsilon))
    loss_eval -= (wdl_eval_target * np.log(wdl_eval_model + epsilon) + (1 - wdl_eval_target) * np.log(
        1 - wdl_eval_model + epsilon))

    loss_result = (game_result * np.log(wdl_eval_target + epsilon) + (1 - game_result) * np.log(
        1 - wdl_eval_target + epsilon))
    loss_result -= (game_result * np.log(wdl_eval_model + epsilon) + (1 - game_result) * np.log(
        1 - wdl_eval_model + epsilon))

    loss = lambda_ * loss_eval + (1 - lambda_) * loss_result

    return loss.mean()


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