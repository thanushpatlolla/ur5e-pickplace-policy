from run_sim import run_sim
import numpy as np

if __name__ == "__main__":
    for i in range (100):
        np.random.seed(i)
        print(i)
        run_sim()

    #53