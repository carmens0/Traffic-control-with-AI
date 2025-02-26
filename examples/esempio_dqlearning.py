from ReinforcementLearning import ExMeteo
from time import sleep  # Per rallentare il rendering
from ReinforcementLearning.deep_q_learning_agent import DQNAgent
import torch
import matplotlib.pyplot as plt
def visualize_agent(agent, environment, n_episodes: int, render: bool = True):
    """Visualizza come l'agente gioca utilizzando la politica appresa."""
    print(f"\n -- Visualizing DQ-agent for {n_episodes} episodes -- ")

    for episode in range(1, n_episodes + 1):
        print(f"Episode {episode} starting.")
        state = environment.reset2(render=True)  # Reset dell'ambiente
        done = False
        score = 0

        while not done:
            action = agent.get_action(state, train = False)  # L'agente prende la decisione basata sui Q-values
            state, reward, terminated, truncated = environment.step2(action)  # Esegui l'azione
            print(f"Weather: {environment.current_weather}, State: {state}, Reward: {reward}")
            score += reward
            # Condizione di fine episodio (sia per terminazione che per troncamento)
            done = terminated or truncated
            sleep(0.1)  # Pausa per migliorare la visibilità del rendering
        avg_wait_time = environment.sim.current_average_wait_time

        print(f"Episode {episode} finished with score {score:.2f} and average wait time {avg_wait_time}")
model_path = r"C:\Users\GENNY C\Desktop\Magistrale Data Science e MAchine Learning\2 anno\1 semestre\AI\Sperimentazione\dq_learning_agent.pkl"
def main():
    # Carica l'ambiente e l'agente
    
    env = ExMeteo()
    state_dim = int(len(env.reset2()))
    action_dim = len(env.action_space)  # La lunghezza di action_space
    dq_agent = DQNAgent(
        state_dim,
        action_dim,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
    )
    dq_agent.model.load_state_dict(torch.load(model_path))
    dq_agent.model.eval()  # Mette il modello in modalità di valutazione (no aggiornamento pesi)

    # Visualizza l'agente in azione (gioco)
    n_episodes = 1  # Numero di episodi da visualizzare
    # Visualizza l'agente in azione (gioco)
    n_episodes = 1  # Numero di episodi da visualizzare
    print("Seleziona quello che vuoi fare: ")
    print("1. Visualizzare un episodio")
    print("2. Valutare il modello")
    choice = input("Seleziona la tua scelta: (1 o 2)")
    if choice == "1":
        visualize_agent(dq_agent, env, n_episodes, render=True)
    if choice == "2":
        n_episodes = input("Seleziona per quanti episodi vuoi valutare il modello: ")
        visualize_agent2(dq_agent, env, int(n_episodes), model_path, render=False)

def visualize_agent2(agent, env, n_test_episodes: int, model_path: str, render: bool = False):
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
            collision_detected += env.sim.collision_detected
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

if __name__ == "__main__":
    main()
