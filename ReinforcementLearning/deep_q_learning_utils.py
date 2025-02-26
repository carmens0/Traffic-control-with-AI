import matplotlib.pyplot as plt
from ReinforcementLearning import ExMeteo
from .deep_q_learning_agent import DQNAgent
import torch

def dq_learning(n_test_episodes):
    env = ExMeteo()
    state_dim = int(len(env.reset2())) # La lunghezza dello stato inizializzato
    action_dim = len(env.action_space)  # La lunghezza di action_space
    dq_agent = DQNAgent(
        state_dim,
        action_dim,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
    )

    n_train_episodes = 300000
    model_save_path = "dq_learning_agent300000.pkl"
    file_name = f"ReinforcementLearning/Traffic_dq_values_{n_train_episodes}.txt"
    train_deep_q_learning(dq_agent, env, file_name, n_train_episodes, render=False, model_save_path=model_save_path ) # addestramento
    validate_model(dq_agent, env, n_test_episodes, model_save_path) # validazione

def save_model(agent, path):
    """Salva il modello di una rete neurale addestrata (PyTorch)."""
    torch.save(agent.q_network.state_dict(), path)  # agent.model è la rete neurale
    print(f"Model saved as {path}")

def train_deep_q_learning(agent, env, path, n_train_episodes: int, render: bool = False, model_save_path: str = "dq_learning_agent.pkl"):
    # Addestramento
    print(f"Training Deep DQ-Learning Agent for {n_train_episodes} episodes...")
    scores = []
    avg_wait_times = []
    for episode in range(1, n_train_episodes + 1):
        state = env.reset2(render=False) # resettiamo l'ambiente ottenendo lo stato iniziale
        score = 0
        done = False
        while not done:
            action = agent.get_action(state) # ottiene l'azione da eseguire
            next_state, reward, done, truncated = env.step2(action) # esegue la simulazione con l'azione scelta ottenendo il nuovo stato e il premio
            if truncated:
                exit()
            agent.update(state, action, reward, next_state, done) # aggiorna i pesi della rete calcolato il target Q e quello previsto con l'MSE
            state = next_state # setta il nuovo stato
            score += reward
        avg_wait_time = env.sim.current_average_wait_time # calcola il tempo medio di attesa per episodio
        avg_wait_times.append(avg_wait_time)
        scores.append(score)    

        if episode % 100 == 0:
            print(f"Episode {episode}: Score = {score}, Average Wait Time = {sum(avg_wait_times)/len(avg_wait_times)}")
    

    save_model(agent, model_save_path)  # Salva l'agente dopo l'addestramento
    print(" -- Training finished -- ")

    # Grafico dei punteggi
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_train_episodes + 1), scores, label='Score per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Training Performance')
    plt.legend()
    plt.grid()

    # Grafico del tempo medio di attesa
    plt.subplot(2, 1, 2)
    plt.plot(range(1, n_train_episodes + 1), avg_wait_times, label='Avg Waiting Time per Episode', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Average Waiting Time (s)')
    plt.title('Average Waiting Time per Episode')
    plt.legend()
    plt.grid()

    # Salvataggio del grafico
    output_file = "training_performance_DQ_learning.jpeg"
    plt.tight_layout()
    plt.savefig(output_file, format='jpeg')
    print(f"Graph saved as {output_file}")

    plt.show()
    print("Deep Q-Learning training complete.")

def validate_model(agent, env, n_test_episodes: int, model_path: str):
    """
    Valuta il modello addestrato su episodi di test.
    """
    # Carica i pesi della rete neurale salvati
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # Mette il modello in modalità di valutazione (no aggiornamento pesi)
    total_collisions, total_wait_time = 0,0
    scores = []
    avg_wait_times = []

    for episode in range(1, n_test_episodes + 1):
        state = env.reset2(render=False)
        score = 0
        done = False
        collision_detected = 0
        while not done:
            action = agent.get_action(state, train=False)  # train=False per disabilitare l'esplorazione
            next_state, reward, done, truncated = env.step2(action)
            if truncated:
                exit()
            state = next_state
            score += reward
            collision_detected += env.sim.collision_detected # controlla se ci sono collisioni
        scores.append(score)
        if collision_detected:
            print(f"Episode {episode} - Collisions: {int(collision_detected)}")
            total_collisions += 1
        else:
            wait_time = env.sim.current_average_wait_time
            total_wait_time += wait_time
            avg_wait_times.append(wait_time)
            print(f"Episode {episode} - Wait time: {wait_time:.2f}")

    # Calcola i risultati finali
    n_completed = n_test_episodes - total_collisions
    print(f"\n -- Results after {n_test_episodes} episodes: -- ")
    print(f"Average wait time per completed episode: {total_wait_time / n_completed:.2f}")
    print(f"Average collisions per episode: {total_collisions / n_test_episodes:.2f}")

    # Grafico degli score
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_test_episodes + 1), scores, label='Score per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Evaluation Performance - Score')
    plt.legend()
    plt.grid()

    # Grafico del tempo medio di attesa
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(avg_wait_times) + 1), avg_wait_times, label='Avg Waiting Time per Episode', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Average Waiting Time (s)')
    plt.title('Evaluation Performance - Average Waiting Time')
    plt.legend()
    plt.grid()

    # Salvataggio dei grafici
    output_file = "evaluation_performance_DQ_Learning.jpeg"
    plt.tight_layout()
    plt.savefig(output_file, format='jpeg')
    print(f"Graph saved as {output_file}")

    plt.show()