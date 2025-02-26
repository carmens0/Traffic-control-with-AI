import random


#La Q-table (tabella dei valori Q) viene costruita dinamicamente mentre l'agente esplora e interagisce con l'ambiente. 
# Non c'è un costruttore esplicito che crea una tabella Q all'inizio del programma. Invece, la Q-table è memorizzata nel dizionario 
# self.q_values, che viene popolato progressivamente tramite il metodo update.

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, actions):
        """ Costruttore inizializza il qlearning con i seguenti parametri: 
        - alpha: tasso di apprendimento, che controlla quanto le nuove informazioni influezano il valore Q
        per una coppia stato-azione. Un valore più alto (vicino a 1) significa che l'agente dà maggiore importanza alle nuove informazioni.
        - espilon: tasso di esplorazione, che controlla la probabilità che l'agente prenda un'azione casuale (esplorando) invece che
        l'azione migliore conosciuta (sfruttando). Se epsilon è 0, l'agente sfrutta sempre la politica migliore.
        - discount: è il fattore di sconto, che determina quanto i premi futuri sono considerati rispetto ai premi immediati: 
        n valore più alto (vicino a 1) significa che l'agente dà maggiore importanza ai premi a lungo termine.
        - actions: è la lista di tutte le azioni possibili che l'agente può compiere nell'ambiente. 
        Memorizza i q-values in un dizionario. """

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(discount)
        self.actions = actions
        self.q_values = {}

    def get_qvalue(self, state, action):
        '''Recupera il valore Q per una data coppia stato-azione dal dizionario q_values.'''
        if (state, action) not in self.q_values: # Se la coppia stato-azione non è ancora presente in q_values, restituisce 0.0 (indicando che non ci sono informazioni precedenti su quella coppia).
            return 0.0
        return self.q_values[(state, action)] # altrimenti, restituisce il valore Q memorizzato per quella coppia stato-azione.


    def get_value(self, state):
        '''calcola il valore massimo dei Q (il miglior valore Q) per un dato stato, 
        esaminando tutte le azioni possibili da quel stato.'''
        action_vals = [self.get_qvalue(state, action) for action in self.actions]
        return max(action_vals)

    def get_policy(self, state):
        """
          calcola la "politica" per un dato stato, ossia trova l'azione 
          migliore da intraprendere in base ai valori Q attuali.
        """
        action_vals = [(action, self.get_qvalue(state, action)) for action in self.actions]
        max_val = max([self.get_qvalue(state, action) for action in self.actions])
        best_actions = [action for action, val in action_vals if val == max_val] 
        return random.choice(best_actions) # se ci sono più azioni migliori scegli una a caso

    def get_action(self, state, train = True):
        """
        Calcola l'azione da intraprendere nello stato attuale.
        Con probabilità self.epsilon, prende un'azione casuale e, 
        altrimenti, prende l'azione migliore secondo la politica. 
        Se non ci sono azioni legali, come nel caso dello stato terminale, sceglie None come azione."
        """
        if train and random.random() < self.epsilon:
            # Esplorazione (scegli un azione a caso)
            return random.choice(self.actions)
        # Sfruttamento (scegli la migliore azione)
        return self.get_policy(state)        

    def update(self, state, action, next_state, reward):
        """
          aggiornare il valore Q della coppia stato-azione dopo 
          che l'agente ha preso un'azione, è passato allo stato successivo
          e ha ricevuto una ricompensa.
        """
        curr_q_val = self.get_qvalue(state, action) # recupera l'attuale valore Q della coppia stato-azione
        self.q_values[(state, action)] = (1 - self.alpha) * curr_q_val + self.alpha * (
                reward + self.discount * self.get_value(next_state))
        # (1 - aplha) * q_corrente + aplha * (reward + discount * q_successivo_max)