import random
import numpy as np
class SARSAAgent:
    def __init__(self, alpha, epsilon, gamma, actions):
        """Inizializza l'agente SARSA."""
        self.alpha = alpha  # Tasso di apprendimento
        self.epsilon = epsilon  # Tasso di esplorazione
        self.gamma = gamma  # Fattore di sconto
        self.actions = actions  # Azioni possibili
        self.q_values = {}  # Memorizza i valori Q come coppie (stato, azione)

    def get_qvalue(self, state, action):
        """Restituisce il valore Q per la coppia stato-azione."""
        if (state, action) not in self.q_values:  # Se la coppia non esiste, restituisce 0.0
            return 0.0
        return self.q_values[(state, action)]

    def get_value(self, state):
        """Restituisce il massimo valore Q per un dato stato."""
        action_vals = [self.get_qvalue(state, action) for action in self.actions]
        return max(action_vals)

    def get_policy(self, state):
        """Restituisce l'azione migliore da intraprendere in base ai valori Q."""
        action_vals = [(action, self.get_qvalue(state, action)) for action in self.actions]
        max_val = max([self.get_qvalue(state, action) for action in self.actions])
        best_actions = [action for action, val in action_vals if val == max_val]
        return random.choice(best_actions)

    def get_action(self, state, train=True):
        """Restituisce l'azione da intraprendere in base alla politica epsilon-greedy."""
        if train and random.random() < self.epsilon:
            return random.choice(self.actions)  # Esplorazione
        return self.get_policy(state)  # Sfruttamento

    def update(self, state, action, next_state, next_action, reward):
        """Aggiorna la coppia stato-azione per SARSA."""
        curr_q_val = self.get_qvalue(state, action) # viene recuperato il Q della coppia stato-azione
        next_q_val = self.get_qvalue(next_state, next_action) # viene recuperato il Q della coppia stato successivo -azione successiva
        self.q_values[(state, action)] = (1 - self.alpha) * curr_q_val + self.alpha * (
            reward + self.gamma * next_q_val)  # Aggiornamento SARSA