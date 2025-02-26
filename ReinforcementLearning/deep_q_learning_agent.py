import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_dim, action_dim, alpha, gamma, epsilon, epsilon_decay=0.99, min_epsilon=0.1):
        self.state_dim = state_dim  # Dimensione dello stato
        self.action_dim = action_dim  # Dimensione dell'azione
        self.alpha = alpha  # Tasso di apprendimento
        self.gamma = gamma  # Fattore di sconto
        self.epsilon = epsilon  # Epsilon per l'esplorazione
        self.epsilon_decay = epsilon_decay  # Decadimento di epsilon
        self.min_epsilon = min_epsilon  # Epsilon minimo

        # Costruzione della rete neurale
        self.q_network = nn.Sequential(
            nn.Linear(self.state_dim, 256),  # il numero di neuroni nel primo hidden layer
            nn.ReLU(),
            nn.Linear(256, 256),  # Secondo strato
            nn.ReLU(),
            nn.Linear(256, self.action_dim)  # Strato di output
        )

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss() # differenza quadratica media tra il Q previsto e il target Q calcolato
        self.model = self.q_network  # Alias per coerenza con il resto del codice

    def get_action(self, state, train = True):
        if train and np.random.rand() < self.epsilon:
            # Esplorazione (azione a caso)
            return np.random.randint(0, self.action_dim)
        else:
            # Sfruttamento (azione con valore Q massimo)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Converti lo stato in tensor
                q_values = self.q_network(state_tensor)  # Ottieni i valori Q
                return torch.argmax(q_values).item()  # Restituisci l'azione con il Q massimo

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Converti stato in tensore
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Converti next_state in tensore
        action_tensor = torch.tensor([action])  # Converti l'azione in tensore
        reward_tensor = torch.tensor([reward])  # Converti il reward in tensore
        done_tensor = torch.tensor([done], dtype=torch.float32)  # Converti done in tensore

        # Predizioni Q
        q_values = self.q_network(state_tensor)  # Valori Q per lo stato corrente
        q_value = q_values[0, action]  # Seleziona il valore Q per l'azione scelta

        # Target Q
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)  # Predizione per il prossimo stato
            max_next_q_value = torch.max(next_q_values).item()  # Ottieni il massimo dei valori Q successivi
            target = torch.tensor([reward + (1 - done_tensor) * self.gamma * max_next_q_value])  # Calcola il target (Bellman)

        # Calcola la perdita e aggiorna la rete
        loss = self.loss_fn(q_value.unsqueeze(0), target)  # tensore di dimensione [1]
        self.optimizer.zero_grad()  # Azzeramento dei gradienti
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Aggiorna i pesi

        # Aggiornamento di epsilon (decadimento dell'esplorazione)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)