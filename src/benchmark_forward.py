import time

import torch
import tqdm

from models.fc_latency import LitFcLatency


def main():
    iterations = 100
    model = LitFcLatency(32, 128, 32, 1.0)
    example_data = torch.randint(0, 1, (32, 16, 1, 32, 32)).to(torch.float)
    start = time.time()
    for _ in tqdm.tqdm(range(iterations)):
        spike_hat, mem_hat = model(example_data)
        pass
    end = time.time()
    print(f"Time elapsed: {end - start}")
    print(f"Time per inference: {(end - start) / iterations}")


if __name__ == "__main__":
    main()
