from .environment2 import ExMeteo
from .sarsa_agents import SARSAAgent 
import matplotlib.pyplot as plt
import pickle

# Hyper-parameters
alpha = 0.3
epsilon = 0.1
discount = 0.9

def save_model(agent, path):
    """Salva l'agente addestrato in un file .pkl"""
    with open(path, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Model saved as {path}")

def save_q_values(path, q_values):
    '''Salva i valori Q dell'apprendimento'''
    with open(path, 'w+') as f:
        f.write(str(q_values))

def get_q_values(path):
    '''Carica i valori Q'''
    q_values_dict = ""
    with open(path, 'r') as f:
        for i in f.readlines():
            q_values_dict = i  # string
    return q_values_dict

def train_agent(agent, environment, path, n_episodes: int, render: bool = False, model_save_path: str = "sarsa_agent.pkl"):
    print(f"\n -- Training SARSA-agent for {n_episodes} episodes  -- ")
    scores = []  # Lista per tenere traccia del punteggio per ogni episodio
    avg_wait_times = []  # Lista per tenere traccia del tempo medio di attesa per episodio
    for n_episode in range(1, n_episodes + 1):
        state = environment.reset(render) # resetta l'ambiente e ottieni lo stato iniziale
        action = agent.get_action(state) # ottieni l'azione da eseguire
        score = 0
        done = False

        while not done:
            new_state, reward, done, truncated = environment.step(action) # esegue la simulazione con l'azione scelta ottenendo un nuovo stato e un premio
            if truncated:
                exit()
            next_action = agent.get_action(new_state)  # Seleziona l'azione successiva basata sulla politica epsilon-greedy
            agent.update(state, action, reward, new_state, next_action)  # Aggiorna il valore Q
            state, action = new_state, next_action  # Imposta il nuovo stato e la nuova azione
            score += reward

        avg_wait_time = environment.sim.current_average_wait_time # funzione che calcola il tempo medio di attesa per l'episodio
        avg_wait_times.append(avg_wait_time)
        scores.append(score)  

    save_q_values(path, agent.q_values) # Salva i valori Q dell'apprendimento
    save_model(agent, model_save_path)  # Salva l'agente dopo l'addestramento
    print(" -- Training finished -- ")

    # Grafico dei punteggi
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_episodes + 1), scores, label='Score per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Training Performance')
    plt.legend()
    plt.grid()

    # Grafico del tempo medio di attesa
    plt.subplot(2, 1, 2)
    plt.plot(range(1, n_episodes + 1), avg_wait_times, label='Avg Waiting Time per Episode', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Average Waiting Time (s)')
    plt.title('Average Waiting Time per Episode')
    plt.legend()
    plt.grid()

    # Salvataggio del grafico
    output_file = "training_performance_SARSA.jpeg"
    plt.tight_layout()
    plt.savefig(output_file, format='jpeg')
    print(f"Graph saved as {output_file}")

    plt.show()

def validate_agent(agent, environment, n_episodes: int, render: bool = False):
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
            action = agent.get_action(state, train = False) # Esegue l'azione migliore in base ai valori Q
            state, reward, done, truncated = environment.step(action) #effettua la simulazione
            if truncated:
                exit()
            score += reward
            collision_detected += environment.sim.collision_detected # funzione per controllare se ci sono state collisioni durante l'episodio

        # Memorizza gli score e i tempi di attesa per il grafico evitando di contare episodi con collisione
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

def load_model(path):
    """Carica un agente dal file .pkl"""
    with open(path, 'rb') as f:
        agent = pickle.load(f)
    print(f"Model loaded from {path}")
    return agent

def sarsa(n_episodes: int, render: bool):
    env: ExMeteo = ExMeteo() # inizializzazione dell' environment
    actions = env.action_space # azioni disponibili
    sarsa_agent = SARSAAgent(alpha, epsilon, discount, actions) # creazione dell'agente
    n_train_episodes = 300000 # numero di episodi di training
    model_save_path = "sarsa_agent.pkl"  # Percorso per il salvataggio del modello
    file_name = f"ReinforcementLearning/Traffic_sarsa_values_{n_train_episodes}.txt" # Percorso per il salvataggio dei valori di Q
    train_agent(sarsa_agent, env, file_name, n_train_episodes, render=False, model_save_path=model_save_path) # addestramento
    sarsa_agent.q_values = eval(get_q_values(file_name)) # aggiungiamo come attributo i valori Q all'agente
    validate_agent(sarsa_agent, env, n_episodes, render) # valutiamo l'addestramento per un totale di n_episodes

