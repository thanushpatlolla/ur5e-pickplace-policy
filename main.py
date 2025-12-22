from run_sim import run_sim
import numpy as np

if __name__ == "__main__":
    np.random.seed(54321)
    for _ in range (100):
        run_sim()
