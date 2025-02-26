# Aggiunta del meteo alla classe environment
import random
from typing import Optional, List, Tuple
from sklearn.preprocessing import OneHotEncoder
from TrafficSimulator import Simulation
from TrafficSimulator.Setups import two_way_intersection_setup

class ExMeteo:
    '''Rappresenta l'ambiente per simulare un'intersezione a due vie con segnali di traffico.'''
    def __init__(self):
        self.action_space: List = [0, 1]   # Le due possibili azioni : semaforo verde per una direzione e rosso per l altro. 
        self.sim: Optional[Simulation] = None #utilizzato per contenere l'oggetto della simulazione, creato tramite la funzione
        self.max_gen: int = 50   # numero massimo di generazioni di veicoli per ogni episodio
        self._vehicles_on_inbound_roads: int = 0 # memorizza il numero di veicoli sulle strade che entrano nell'interseazione
        self.current_weather: str = "Rain" #inizializzazione del meteo in Pioggia
        self.weather_conditions: List[str] = ["Sunny", "Rain", "Snow", "Fog"] # lista dei meteo possibili

    def update_weather(self):
        """Aggiorna casualmente le condizioni meteorologiche e sincronizza con la simulazione."""
        self.current_weather = random.choice(self.weather_conditions) # Si seleziona casualmente un meteo dalla lista
        if self.sim:
            self.sim.set_weather(self.current_weather)  # Aggiorna il meteo nella simulazione
            # Aggiorna il comportamento dei veicoli basandosi sul meteo
            for road in self.sim.roads:
                for vehicle in road.vehicles:
                    vehicle.update_behavior_based_on_weather(self.current_weather)


    def step(self, step_action) -> Tuple[Tuple, float, bool, bool]:
        self.sim.run(step_action)  # esegue la simulazione di un'azione scelta (0 o 1)
        # Aggiorna il meteo con una probabilità del 10% e sincronizza con la simulazione
        if random.random() < 0.1:
            self.update_weather()
        new_state: Tuple = self.get_state()  # Chiama il metodo get_state per ottenere lo stato aggiornato
        step_reward: float = self.get_reward(new_state)
        # Aggiorna il numero di veicoli sulle strade in entrata
        n_west_east_vehicles, n_south_north_vehicles = new_state[1], new_state[2]
        self._vehicles_on_inbound_roads = n_west_east_vehicles + n_south_north_vehicles
        terminated: bool = self.sim.completed
        truncated: bool = self.sim.gui_closed
        return new_state, step_reward, terminated, truncated
        
        
    def get_state(self) -> Tuple:
        '''Lo stato è una tupla che contiene: 
        1. Stato del segnale del traffico : rappresenta la fase attuale del ciclo del segnale (verde in una direzione e rosso nell'altra). 
        2. Il numero di veicoli in ogni direzione: somma dei veicoli sulle strade in ingresso all'intersezione. 
        3. Indicazione di intersezione vuota : calcola il numero di veicoli sulla mappa e sottrae : i veicoli sulla strada in uscita, i veicoli in ingresso; se 
        questo risultato >0 allora l'intersezione non è vuota. 
        4. Il meteo in atto. '''
        state = []
        for traffic_signal in self.sim.traffic_signals:
            junction = []
            traffic_signal_state = traffic_signal.current_cycle[0]
            junction.append(traffic_signal_state)

            for direction in traffic_signal.roads:
                junction.append(sum(len(road.vehicles) for road in direction))

                # Calcola il tempo medio di attesa per ciascuna direzione
            avg_wait_times = []
            for direction in traffic_signal.roads:
                total_wait_time = 0
                total_vehicles = 0
                for road in direction:
                    for vehicle in road.vehicles:
                        total_wait_time += vehicle.get_wait_time(self.sim.t)  # Ottieni il tempo di attesa del veicolo
                        total_vehicles += 1
            # Calcola il tempo medio di attesa per la direzione
                #avg_wait_times.append(total_wait_time / total_vehicles if total_vehicles > 0 else 0)
            #junction.extend(avg_wait_times)
            
            n_direction_1_vehicles, n_direction_2_vehicles = junction[1], junction[2]
            out_bound_vehicles = sum(len(self.sim.roads[i].vehicles) for i in self.sim.outbound_roads)
            non_empty_junction = bool(self.sim.n_vehicles_on_map - out_bound_vehicles -
                                    n_direction_1_vehicles - n_direction_2_vehicles)
            junction.append(non_empty_junction)
            state.append(junction)

        state = state[0]

        return tuple(state) + (self.current_weather,)


    def get_reward(self, state: Tuple) -> float:

        traffic_signal_state, n_direction_1_vehicles, n_direction_2_vehicles,  non_empty_junction, weather = state
  
        # wait_time_penalty = 0
        # wait_times = self.sim.get_wait_times()  # Ottieni i tempi di attesa di tutti i veicoli
        # for wait_time in wait_times:
        #     wait_time_penalty -= wait_time * 0.5  # Penalità proporzionale al tempo di attesa

        # Penalità per collisioni
        #collision_penalty = -20 if self.sim.collision_detected else 0

        flow_change = self._vehicles_on_inbound_roads - n_direction_1_vehicles - n_direction_2_vehicles
        #average_wait_time = self.sim.current_average_wait_time
        average_wait_time_on_map = self.sim.current_average_wait_time_on_map
        # Ricompensa totale
        total_reward = ( flow_change #- round(average_wait_time_on_map)
                    )

        return total_reward



    def reset(self, render=False) -> Tuple:
        self.sim = two_way_intersection_setup(self.max_gen)
        if render:
            self.sim.init_gui()
        self._vehicles_on_inbound_roads = 0  # Reset del contatore

        # Aggiorna il meteo all'inizio di un episodio
        self.update_weather()

        # Aggiorna il comportamento dei veicoli in base al meteo iniziale
        for road in self.sim.roads:
            for vehicle in road.vehicles:
                vehicle.update_behavior_based_on_weather(self.current_weather)

        init_state = self.get_state()
        return init_state
    
    # PER DEEP Q LEARNING
    #Siccome includiamo il meteo nello stato, trasformiamo il meteo con il metodo one hot
    weather_categories = ["Sunny", "Rain", "Snow", "Fog"]
    weather_encoder = OneHotEncoder(categories=[weather_categories], sparse_output=False)

    def preprocess_state(self, state: Tuple) -> list:
        traffic_signal_state = int(state[0])  # Converti booleano in intero
        vehicles_in_directions = state[1:3]  # Veicoli direzionali (già numerici)
        non_empty_junction = int(state[3])  # Converti booleano in intero
        weather = state[4]  # Stringa del meteo

        # Applica il one-hot encoding al meteo
        weather_encoded = self.weather_encoder.fit_transform([[weather]])[0]

        # Combina tutte le informazioni preprocessate
        return [traffic_signal_state] + list(vehicles_in_directions) + [non_empty_junction] + weather_encoded.tolist()


    def get_state2(self) -> Tuple:
        state = []
        for traffic_signal in self.sim.traffic_signals:
            junction = []
            traffic_signal_state = traffic_signal.current_cycle[0]
            junction.append(traffic_signal_state)

            for direction in traffic_signal.roads:
                junction.append(sum(len(road.vehicles) for road in direction))

            n_direction_1_vehicles, n_direction_2_vehicles = junction[1], junction[2]
            out_bound_vehicles = sum(len(self.sim.roads[i].vehicles) for i in self.sim.outbound_roads)
            non_empty_junction = bool(self.sim.n_vehicles_on_map - out_bound_vehicles - n_direction_1_vehicles - n_direction_2_vehicles)
            junction.append(non_empty_junction)
            state.append(junction)

        state = state[0]

        # Codifica One-Hot per il meteo
        weather_onehot = self.weather_encoder.fit_transform([[self.current_weather]])

        # Restituisci stato + meteo codificato
        return tuple(state) + tuple(weather_onehot[0])  # Concatenare il meteo codificato

    def reset2(self, render=False) -> list:
        self.sim = two_way_intersection_setup(self.max_gen)
        if render:
            self.sim.init_gui()
        self._vehicles_on_inbound_roads = 0  # Reset del contatore

        # Aggiorna il meteo all'inizio di un episodio
        self.update_weather()

        # Aggiorna il comportamento dei veicoli in base al meteo iniziale
        for road in self.sim.roads:
            for vehicle in road.vehicles:
                vehicle.update_behavior_based_on_weather(self.current_weather)

        # Restituisci lo stato preprocessato
        return self.get_state2()
    
    def step2(self, step_action) -> Tuple[list, float, bool, bool]:
        """
        Esegue un passo nella simulazione con l'azione scelta.
        Restituisce lo stato preprocessato, la ricompensa, e i flag di terminazione/troncamento.
        """
        self.sim.run(step_action)  # Esegue la simulazione dell'azione scelta (0 o 1)

        # Aggiorna il meteo con una probabilità del 10% e sincronizza con la simulazione
        if random.random() < 0.1:
            self.update_weather()

        # Ottieni lo stato preprocessato tramite il metodo get_state
        new_state: list = self.get_state2()

        # Ottieni la ricompensa
        step_reward: float = self.get_reward2(new_state)

        # Aggiorna il numero di veicoli sulle strade in entrata
        n_west_east_vehicles, n_south_north_vehicles = new_state[1], new_state[2]
        self._vehicles_on_inbound_roads = n_west_east_vehicles + n_south_north_vehicles

        # Flag di terminazione
        terminated: bool = self.sim.completed
        truncated: bool = self.sim.gui_closed

        return new_state, step_reward, terminated, truncated
    

    def get_reward2(self, state: Tuple) -> float:
        traffic_signal_state, n_direction_1_vehicles, n_direction_2_vehicles, non_empty_junction, weather = state[:5]
        weather_onehot = state[5:]


        # wait_time_penalty = 0
        # wait_times = self.sim.get_wait_times()  # Ottieni i tempi di attesa di tutti i veicoli
        # for wait_time in wait_times:
        #     wait_time_penalty -= wait_time * 0.5  # Penalità proporzionale al tempo di attesa

        # Penalità uniforme per collisioni
        #collision_penalty = -20 if self.sim.collision_detected else 0

        flow_change = self._vehicles_on_inbound_roads - n_direction_1_vehicles - n_direction_2_vehicles
        average_wait_time_on_map = self.sim.current_average_wait_time_on_map

        # Ricompensa totale
        total_reward = ( flow_change - round(average_wait_time_on_map)
                    )

        return total_reward