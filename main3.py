from ReinforcementLearning.environment2 import ExMeteo  # importa ExMeteo direttamente
import torch
from ReinforcementLearning.deep_q_learning_agent import DQNAgent
from ReinforcementLearning.q_learning_agent import QLearningAgent
from ReinforcementLearning.sarsa_agents import SARSAAgent
t = 30  # Cycle time threshold

def get_q_values(path):
    '''Carica i valori Q'''
    q_values_dict = ""
    with open(path, 'r') as f:
        for i in f.readlines():
            q_values_dict = i  # string
    return q_values_dict

# Azione a ciclo fisso
def fixed_cycle_action(sim, dummy=None) -> bool:
    """Returns a boolean indicating whether to take an action
    if enough time has elapsed since the previous action."""
    switch = False
    traffic_signal = sim.traffic_signals[0]
    time_elapsed = sim.t - traffic_signal.prev_update_time >= t
    if time_elapsed:
        traffic_signal.prev_update_time = sim.t
        switch = True
    return switch


# Azione basata sulla coda più lunga
def longest_queue_action(curr_state, prev_state) -> bool:
    """Returns a boolean indicating whether to take an action
    if enough time has elapsed since the previous action."""
    switch = False
    traffic_signal = curr_state.traffic_signals[0]
    time_elapsed = curr_state.t - traffic_signal.prev_update_time >= t
    if time_elapsed:
        traffic_signal_state, n_direction_1_vehicles, n_direction_2_vehicles, non_empty_junction, weather = prev_state
        # Se la direzione con più veicoli ha un semaforo rosso, cambialo in verde
        if traffic_signal_state and n_direction_1_vehicles < n_direction_2_vehicles:
            switch = True
        elif not traffic_signal_state and n_direction_1_vehicles > n_direction_2_vehicles:
            switch = True
    if switch:
        # Aggiorna il tempo di aggiornamento del semaforo
        traffic_signal.prev_update_time = curr_state.t
    return switch


# Funzioni d'azione disponibili
action_funcs = {'fc': fixed_cycle_action, 'lqf': longest_queue_action}


# Funzione ciclo predefinito
def default_cycle(n_episodes: int, action_func_name: str, render):
    print(f"\n -- Running {action_func_name.upper()} for {n_episodes} episodes  -- ")
    environment: ExMeteo = ExMeteo()
    total_wait_time, total_collisions = 0, 0
    action_func = action_funcs[action_func_name]
    
    for episode in range(1, n_episodes + 1):
        state = environment.reset(render)
        score = 0
        collision_detected = 0
        done = False

        while not done:
            action = action_func(environment.sim, state)
            state, reward, done, truncated = environment.step(action)
            if truncated:
                exit()
            score += reward
            collision_detected += environment.sim.collision_detected

        if collision_detected:
            print(f"Episode {episode} - Collisions: {int(collision_detected)}")
            total_collisions += 1
        else:
            wait_time = environment.sim.current_average_wait_time
            total_wait_time += wait_time
            print(f"Episode {episode} - Wait time: {wait_time:.2f}")

    n_completed = n_episodes - total_collisions
    print(f"\n -- Results after {n_episodes} episodes: -- ")
    print(f"Average wait time per completed episode: {total_wait_time / n_completed:.2f}")
    print(f"Average collisions per episode: {total_collisions / n_episodes:.2f}")
    total_wait_time_lqf = total_wait_time / n_completed
    return total_wait_time_lqf
def visualize_agent2(agent, env, n_test_episodes: int, model_path: str, render: bool):
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
        state = env.reset2(render)
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

def visualize_agent_q(agent, environment, n_episodes: int, render: bool):
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

import matplotlib.pyplot as plt

def validate_models(n_episodes=1000, render=False):
    """Valida tutti i modelli e genera un grafico comparativo."""
    env = ExMeteo()
    results = {}

    # Fixed Cycle
    print("\nValidating Fixed Cycle...")
    total_wait_time_fc, _ = 0, 0
    for _ in range(n_episodes):
        state = env.reset(render)
        done = False
        while not done:
            action = fixed_cycle_action(env.sim)
            state, _, done, _ = env.step(action)
        total_wait_time_fc += env.sim.current_average_wait_time
    results["Fixed Cycle"] = total_wait_time_fc / n_episodes

    # Longest Queue First
    print("\nValidating Longest Queue First...")
    results["Longest Queque First"] = default_cycle(n_episodes, action_func_name='lqf', render=render)


    # Deep Q-Learning
    print("\nValidating Deep Q-Learning...")
    dq_agent = DQNAgent(len(env.reset2()), len(env.action_space), alpha=0.001, gamma=0.99, epsilon=1.0)
    model_path = r"C:\Users\GENNY C\Desktop\Magistrale Data Science e MAchine Learning\2 anno\1 semestre\AI\Sperimentazione\dq_learning_agent.pkl"
    dq_agent.model.load_state_dict(torch.load(model_path))
    dq_agent.model.eval()
    total_wait_time_dq = 0
    for _ in range(n_episodes):
        state = env.reset2(render)
        done = False
        while not done:
            action = dq_agent.get_action(state, train=False)
            state, _, done, _ = env.step2(action)
        total_wait_time_dq += env.sim.current_average_wait_time
    results["Deep Q-Learning"] = total_wait_time_dq / n_episodes

    # Q-Learning
    print("\nValidating Q-Learning...")
    q_agent = QLearningAgent(0.1, 0.1, 0.6, env.action_space)
    q_agent.q_values = eval(get_q_values("ReinforcementLearning/Traffic_q_values_48000.txt"))
    total_wait_time_q = 0
    for _ in range(n_episodes):
        state = env.reset(render)
        done = False
        while not done:
            action = q_agent.get_action(state, train=False)
            state, _, done, _ = env.step(action)
        total_wait_time_q += env.sim.current_average_wait_time
    results["Q-Learning"] = total_wait_time_q / n_episodes

    # SARSA
    print("\nValidating SARSA...")
    sarsa_agent = SARSAAgent(0.1, 0.1, 0.6, env.action_space)
    sarsa_agent.q_values = eval(get_q_values("ReinforcementLearning/Traffic_sarsa_values_300000.txt"))
    total_wait_time_sarsa = 0
    for _ in range(n_episodes):
        state = env.reset(render)
        done = False
        while not done:
            action = sarsa_agent.get_action(state, train=False)
            state, _, done, _ = env.step(action)
        total_wait_time_sarsa += env.sim.current_average_wait_time
    results["SARSA"] = total_wait_time_sarsa / n_episodes

    # Visualizza i risultati
    print("\nResults:", results)
    models = list(results.keys())
    avg_wait_times = list(results.values())
    mean_wait_time = sum(avg_wait_times) / len(avg_wait_times)

    plt.figure(figsize=(10, 6))
    plt.bar(models, avg_wait_times, color='skyblue', label='Avg Wait Time')
    plt.axhline(mean_wait_time, color='red', linestyle='--', label='Mean Wait Time')
    plt.xlabel('Models')
    plt.ylabel('Average Wait Time')
    plt.title('Model Comparison: Average Wait Time')
    plt.legend()
    plt.show()




# Menu principale
def main_menu():
    while True:
        print("\n*** Traffic Light Controller ***")
        print("1. Fixed Cycle (FC)")
        print("2. Longest Queue First (LQF)")
        print("3. Deep Q-Learning")
        print("4. Q-Learning")
        print("5. Sarsa")
        print("6. Validate Models")
        print("0. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            episodes = int(input("Enter number of episodes: "))
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            default_cycle(n_episodes=episodes, action_func_name='fc', render=render)
        
        elif choice == "2":
            episodes = int(input("Enter number of episodes: "))
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            default_cycle(n_episodes=episodes, action_func_name='lqf', render=render)
        
        elif choice == "3":
            model_path = r"C:\Users\GENNY C\Desktop\Magistrale Data Science e MAchine Learning\2 anno\1 semestre\AI\Sperimentazione\dq_learning_agent.pkl"
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
            n_episodes = input("Seleziona per quanti episodi vuoi valutare il modello: ")
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            visualize_agent2(dq_agent, env, int(n_episodes), model_path, render)
        elif choice == "4":
            env = ExMeteo()
            q_agent = QLearningAgent(alpha=0.1, epsilon=0.1, discount=0.6, actions=env.action_space)
            # Carica i valori Q pre-addestrati
            q_agent.q_values = eval(get_q_values("ReinforcementLearning/Traffic_q_values_48000.txt"))
            n_episodes = input("Seleziona per quanti episodi vuoi valutare il modello: ")
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            visualize_agent_q(q_agent, env, int(n_episodes), render)
        elif choice == "5":
            env = ExMeteo()
            q_agent = SARSAAgent(alpha=0.1, epsilon=0.1, gamma=0.6, actions=env.action_space)
            q_agent.q_values = eval(get_q_values("ReinforcementLearning/Traffic_sarsa_values_300000.txt"))
            n_episodes = input("Seleziona per quanti episodi vuoi valutare il modello: ")
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            visualize_agent_q(q_agent, env, int(n_episodes), render)
        elif choice == "6":
            render = input("Render simulation? (yes/no): ").lower() == 'yes'
            validate_models(n_episodes=1000, render=render)
        elif choice == "0":
            print("Exiting... Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


# Avvio del programma
if __name__ == "__main__":
    main_menu()
