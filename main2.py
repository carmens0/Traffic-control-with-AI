import torch
import os
from ReinforcementLearning import q_learning, dq_learning, sarsa



def main():
    print("Select an agent to train:")
    print("1. Q-Learning Agent")
    print("2. Deep Q-Learning Agent")
    print("3. SARSA")
    choice = input("Enter your choice (1, 2 or 3): ")

    if choice == "1":
        q_learning(1000, False)
    elif choice == "2":
        dq_learning(1000)
    elif choice == "3":
        sarsa(1000, False)
    else:
        print("Invalid choice. Please select 1, 2 or 3.")

if __name__ == "__main__":
    main()
