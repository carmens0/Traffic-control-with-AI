from ReinforcementLearning import ExMeteo, load_model, get_q_values
import matplotlib.pyplot as plt
from time import sleep  # Per rallentare il rendering
from ReinforcementLearning.sarsa_agents import SARSAAgent

def visualize_agent(agent, environment, n_episodes: int, render: bool = True):
    """Visualizza come l'agente gioca utilizzando la politica appresa."""
    print(f"\n -- Visualizing Q-agent for {n_episodes} episodes -- ")

    for episode in range(1, n_episodes + 1):
        print(f"Episode {episode} starting.")
        state = environment.reset(render=True)  # Reset dell'ambiente
        done = False
        score = 0
        avg_wait_time = 0
        while not done:
            action = agent.get_action(state, train = False)  # L'agente prende la decisione basata sui Q-values
            state, reward, terminated, truncated = environment.step(action)  # Esegui l'azione

            print(f"Weather: {environment.current_weather}, State: {state}, Reward: {reward}")
            score += reward
            # Condizione di fine episodio (sia per terminazione che per troncamento)
            done = terminated or truncated
            sleep(0.1)  # Pausa per migliorare la visibilità del rendering
        avg_wait_time = environment.sim.current_average_wait_time
        print(f"Episode {episode} finished with score {score:.2f} and average wait time {avg_wait_time} ")
        
def visualize_agent2(agent, environment, n_episodes: int, render: bool = False):
    '''Valuta l'agente eseguendo episodi di validazione senza aggiornare i valori Q. 
    Monitora il tempo di attesa medio e il numero di collisioni. '''
    print(f"\n -- Evaluating Q-agent for {n_episodes} episodes -- ")
    total_wait_time, total_collisions, n_completed = 0, 0, 0
    scores = []  # Lista per memorizzare gli score per episodio
    avg_wait_times = []  # Lista per memorizzare il tempo medio di attesa per episodio

    for episode in range(1, n_episodes + 1):
        state = environment.reset(render)
        score = 0
        collision_detected = 0
        done = False

        while not done:
            action = agent.get_action(state, train = False)
            state, reward, done, truncated = environment.step(action)
            if truncated:
                exit()
            score += reward
            collision_detected += environment.sim.collision_detected

        # Memorizza gli score e i tempi di attesa per il grafico
        scores.append(score)
        if collision_detected:
            print(f"Episode {episode} - Collisions: {int(collision_detected)}")
            total_collisions += 1
        else:
            wait_time = environment.sim.current_average_wait_time
            total_wait_time += wait_time
            avg_wait_times.append(wait_time)
            print(f"Episode {episode} - Wait time: {wait_time:.2f}")

    # Calcola i risultati finali
    n_completed = n_episodes - total_collisions
    print(f"\n -- Results after {n_episodes} episodes: -- ")
    print(f"Average wait time per completed episode: {total_wait_time / n_completed:.2f}")
    print(f"Average collisions per episode: {total_collisions / n_episodes:.2f}")

    # Grafico degli score
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_episodes + 1), scores, label='Score per Episode')
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
    output_file = "evaluation_performance_SARSA.jpeg"
    plt.tight_layout()
    plt.savefig(output_file, format='jpeg')
    print(f"Graph saved as {output_file}")

    plt.show()

def main():
    # Carica l'ambiente e l'agente
    env = ExMeteo()
    q_agent = SARSAAgent(alpha=0.1, epsilon=0.1, gamma=0.6, actions=env.action_space)

    # Carica i valori Q pre-addestrati
    # (Assumiamo che tu abbia già addestrato l'agente e salvato i valori Q in un file)
    q_agent.q_values = eval(get_q_values("ReinforcementLearning/Traffic_sarsa_values_300000.txt"))
    # Visualizza l'agente in azione (gioco)
    n_episodes = 1  # Numero di episodi da visualizzare
    print("Seleziona quello che vuoi fare: ")
    print("1. Visualizzare un episodio")
    print("2. Valutare il modello")
    choice = input("Seleziona la tua scelta: (1 o 2)")
    if choice == "1":
        visualize_agent(q_agent, env, n_episodes, render=True)
    if choice == "2":
        n_episodes = input("Seleziona per quanti episodi vuoi valutare il modello: ")
        visualize_agent2(q_agent, env, int(n_episodes), render=False)
if __name__ == "__main__":
    main()
