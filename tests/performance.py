import random
import time

import torch
import torch.ao.quantization as quantization


from simple_nnue import model

# number of inputs per second
def test_performance():
    # Create a model
    nnue = model.NNUE()

    # Create some random inputs
    features = torch.randint(0, 127, (10000, 2, model.NUM_FEATURES)).float()

    # timer variable
    timer = 0

    for i in range(features.shape[0]):
        white_features = features[i, 0, :]
        black_features = features[i, 1, :]
        stm = random.randrange(2)

        # start timer
        start = time.time()

        # Make a forward pass
        nnue.forward(white_features, black_features, stm)

        # end timer
        end = time.time()

        # add time to timer
        timer += end - start

    print(f"Time taken for {features.shape[0]} forward passes: {timer} seconds")

def test_performance_quantization():
    batch_size = 1000

    # Create a model
    nnue = model.NNUE()

    nnue.qconfig = quantization.default_qconfig
    quantization.prepare(nnue, inplace=True)

    calibration_data = torch.randint(0, 128, (10000, 2, model.NUM_FEATURES)).float()

    for i in range(0, calibration_data.shape[0], batch_size):
        # batch_features = calibration_data[i:i + batch_size]
        # white_features = batch_features[:, 0, :]
        # black_features = batch_features[:, 1, :]

        white_features = torch.randint(0, 127, (batch_size, model.NUM_FEATURES)).float()
        black_features = torch.randint(0, 127, (batch_size, model.NUM_FEATURES)).float()

        stms = torch.randint(0, 2, (batch_size, 1))

        nnue.forward(white_features, black_features, stms)

    quantization.convert(nnue, inplace=True)

    # Create some random inputs (1 mil)
    features = torch.randint(0, 128, (10000, 2, model.NUM_FEATURES)).float()

    # timer variable
    timer = 0

    for i in range(features.shape[0]):
        white_features = features[i, 0, :].unsqueeze(0)
        black_features = features[i, 1, :].unsqueeze(0)
        stm = random.randrange(2)

        # start timer
        start = time.time()

        # Make a forward pass
        nnue.forward(white_features, black_features, stm)

        # end timer
        end = time.time()

        # add time to timer
        timer += end - start

    print(f"Time taken for {features.shape[0]} forward passes: {timer} seconds\t (Quantized model)")


if __name__ == "__main__":
    # test_performance()
    test_performance_quantization()