"""
 * Name: Nima Jamshidi
 * Professor Leilani Gilpin
 * AIEA (Task 8) PPO Improvements
 * CMPM-118
 * June 2nd, 2026
"""

import subprocess
import time
import os

LOG_DIR = "runs/task9_benchmark"
ALGORITHMS = {"DQN": "Improved_DQN_train.py",
              "PPO": "Improved_PPO_train.py"}

def run_algorithm(name, file_name):
    return subprocess.Popen(["python", file_name])

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    processes = []

    for name, filename in ALGORITHMS.items():
        process = run_algorithm(name, filename)
        processes.append((name, process))
        time.sleep(2)

    print("PPO and DQN are training")
    print(f"tensorboard --logdir={LOG_DIR} --host=0.0.0.0 --port=6006")
    
    for name, process in processes:
        process.wait()
        print(f"{name} is done")

        print("\nAll benchmarks finished.")

if __name__ == "__main__":
    main()



    



  