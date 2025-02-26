import numpy as np
import pygame
from pygame.draw import polygon


# # For debugging purposes
# DRAW_VEHICLE_IDS = True
# DRAW_ROAD_IDS = False
# FILL_POLYGONS = True



import pygame
import numpy as np

class SunnyEffect:
    def __init__(self, intensity=100):
        # Intensità della luce solare, variabile da 0 a 255
        self.intensity = intensity

    def apply(self, screen):
        """Applica l'effetto di luce solare sopra lo schermo"""
        # Colore del cielo soleggiato: azzurro chiaro
        sky_color = (135, 206, 235)  # Azzurro chiaro (simula il cielo sereno)
        
        # Crea una superficie che rappresenta il cielo soleggiato
        # La superficie va a coprire tutto lo schermo, con una leggera luminosità
        screen.fill(sky_color)  # Riempie lo schermo con il colore del cielo soleggiato
        
        # Aggiungi una sorta di "bagliore" attorno ai bordi (per imitare il sole)
        # Usa un "overlay" semitrasparente di colore giallo chiaro
        glow_color = (255, 255, 204)  # Giallo chiaro (simula il bagliore del sole)
        
        # Disegna un "bagliore" sfumato sui bordi
        glow_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        glow_surface.fill((255, 255, 204, self.intensity))  # Giallo con intensità di opacità
        
        # Sovrapponi il bagliore alla scena
        screen.blit(glow_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)  # Usa un blending per il bagliore

class FogEffect:
    def __init__(self, opacity=100):
        # Opacità iniziale della nebbia (completamente trasparente 0, completamente opaco 255)
        self.opacity = opacity
        self._fog_surface = None

    def _create_fog_surface(self, width, height):
        """Crea una superficie trasparente che simula la nebbia"""
        # Crea una superficie che copre l'intero schermo con un colore bianco trasparente
        self._fog_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._fog_surface.fill((255, 255, 255, self.opacity))  # Colore bianco con opacità

    def update(self, screen):
        """Applica la nebbia sopra lo schermo"""
        # Se la superficie della nebbia non è stata creata, la creiamo
        if self._fog_surface is None:
            self._create_fog_surface(screen.get_width(), screen.get_height())
        
        # Disegna la superficie della nebbia sopra lo schermo
        screen.blit(self._fog_surface, (0, 0))  # Sovrappone la nebbia sopra tutto lo schermo

class RainEffect:
    def __init__(self, num_drops=100, max_speed=5):
        # Numero di gocce di pioggia e la velocità massima delle gocce
        self.num_drops = num_drops
        self.max_speed = max_speed
        # Creazione di gocce di pioggia con posizioni casuali e velocità variabili
        self.drops = [
            {"x": np.random.randint(0, 800),  # x casuale
             "y": np.random.randint(0, 600),  # y casuale
             "speed": np.random.uniform(1.0, max_speed)}  # Velocità casuale tra 1.0 e max_speed
            for _ in range(self.num_drops)
        ]

    def _move_drops(self):
        """Muove le gocce di pioggia verso il basso con la velocità e ricicla quelle che escono dallo schermo."""
        for drop in self.drops:
            # Muove la goccia verso il basso in base alla sua velocità
            drop["y"] += drop["speed"]

            # Se la goccia esce dal limite inferiore dello schermo, la ricicla in alto
            if drop["y"] > 600:  # Supponiamo che l'altezza dello schermo sia 600
                drop["y"] = 0
                drop["x"] = np.random.randint(0, 800)  # Ricrea una nuova posizione x casuale

    def _draw_rain(self, screen):
        """Disegna le gocce di pioggia sullo schermo."""
        for drop in self.drops:
            pygame.draw.line(screen, (0, 0, 255), (drop["x"], drop["y"]), (drop["x"], drop["y"] + 10), 2)

    def update(self, screen):
        """Aggiorna la posizione delle gocce e le disegna sullo schermo."""
        self._move_drops()  # Muove le gocce
        self._draw_rain(screen)  # Disegna le gocce

class SnowEffect:
    def __init__(self, num_flakes=200):
        self.num_flakes = num_flakes
        self.snowflakes = [
            {"x": np.random.randint(0, 800),  # x casuale
             "y": np.random.randint(0, 600),  # y casuale
             "speed": np.random.uniform(0.5, 3.0)}  # Velocità casuale tra 0.5 e 3.0
            for _ in range(self.num_flakes)
        ]
        self.snow_on_roads = []

    def _move_snowflakes(self):
        """Muove i fiocchi di neve e li ricicla se superano il bordo inferiore."""
        for flake in self.snowflakes:
            flake["y"] += flake["speed"]

            if flake["y"] > 600:
                flake["y"] = 0
                flake["x"] = np.random.randint(0, 800)  # Ricrea una nuova posizione x casuale

    def _check_snow_on_roads(self, roads):
        """Verifica se un fiocco è sopra una strada e accumula neve."""
        for flake in self.snowflakes:
            for road in roads:
                # Verifica se il fiocco è sopra la strada (approssimazione)
                road_start_x, road_start_y = road.start
                road_end_x, road_end_y = road.end
                road_width = 3.7  # Larghezza della strada (approssimazione)
                # Calcoliamo la proiezione del fiocco sulla strada
                if road_start_x <= flake["x"] <= road_end_x and road_start_y - road_width <= flake["y"] <= road_start_y:
                    self.snow_on_roads.append((flake["x"], flake["y"]))

    def _draw_snow(self, screen):
        """Disegna i fiocchi di neve sullo schermo."""
        for flake in self.snowflakes:
            pygame.draw.circle(screen, (255, 255, 255), (flake["x"], int(flake["y"])), 2)

    def _draw_snow_on_roads(self, screen):
        """Disegna la neve accumulata sulle strade."""
        for snow in self.snow_on_roads:
            x, y = snow
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 3)  # Rappresenta l'accumulo della neve

    def update(self, screen, roads):
        """Aggiorna la posizione dei fiocchi di neve, accumula neve sulle strade e disegna tutto."""
        self._move_snowflakes()
        self._check_snow_on_roads(roads)
        self._draw_snow(screen)
        self._draw_snow_on_roads(screen)

class Window:
    def __init__(self, simulation):
        self._width = 1000
        self._height = 630
        self.closed: bool = False
        self._sim = simulation
        self._background_color = (235, 235, 235)
        self._screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption('AI Traffic Lights Controller')
        pygame.display.flip()
        pygame.font.init()
        font = f'Lucida Console'
        self._text_font = pygame.font.SysFont(font, 16)
        self._zoom = 5
        self._offset = (0, 0)
        self._mouse_last = (0, 0)
        self._mouse_down = False
        self.weather_visual = None  # Placeholder per l'elemento grafico che rappresenta il meteo
        self.snow_effect = SnowEffect()  # La classe SnowEffect non richiede lo screen
        self.rain_effect=RainEffect()
        self.fog_effect=FogEffect()
        self.sunny_effect=SunnyEffect()

   

    def update(self) -> None:
        self._draw()
        self._weather = self._sim.get_weather()

        pygame.display.update()
        for event in pygame.event.get():
            # Quit program if window is closed
            if event.type == pygame.QUIT:
                self.closed = True
            # Handle mouse events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If mouse button down
                if event.button == pygame.BUTTON_LEFT:
                    # Left click
                    x, y = pygame.mouse.get_pos()
                    x0, y0 = self._offset
                    self._mouse_last = (x - x0 * self._zoom, y - y0 * self._zoom)
                    self._mouse_down = True
                if event.button == pygame.BUTTON_WHEELUP:
                    # Mouse wheel up
                    self._zoom *= (self._zoom ** 2 + self._zoom / 4 + 1) / (self._zoom ** 2 + 1)
                if event.button == pygame.BUTTON_WHEELDOWN:
                    # Mouse wheel down
                    self._zoom *= (self._zoom ** 2 + 1) / (self._zoom ** 2 + self._zoom / 4 + 1)
            elif event.type == pygame.MOUSEMOTION:
                # Drag content
                if self._mouse_down:
                    x1, y1 = self._mouse_last
                    x2, y2 = pygame.mouse.get_pos()
                    self._offset = ((x2 - x1) / self._zoom, (y2 - y1) / self._zoom)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_down = False

    def _convert(self, x, y=None):
        """Converts simulation coordinates to screen coordinates"""
        if isinstance(x, list):
            return [self._convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._convert(*x)
        return (int(self._width / 2 + (x + self._offset[0]) * self._zoom),
                int(self._height / 2 + (y + self._offset[1]) * self._zoom))

    def _inverse_convert(self, x, y=None):
        """Converts screen coordinates to simulation coordinates"""
        if isinstance(x, list):
            return [self._convert(e[0], e[1]) for e in x]
        if isinstance(x, tuple):
            return self._convert(*x)
        return (int(-self._offset[0] + (x - self._width / 2) / self._zoom),
                int(-self._offset[1] + (y - self._height / 2) / self._zoom))

    def _rotated_box(self, pos, size, angle=None, cos=None, sin=None, centered=True,
                     color=(0, 0, 255)):
        """Draws a rectangle center at *pos* with size *size* rotated anti-clockwise by *angle*."""

        def vertex(e1, e2):
            return (x + (e1 * l * cos + e2 * h * sin) / 2,
                    y + (e1 * l * sin - e2 * h * cos) / 2)

        x, y = pos
        l, h = size
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        if centered:
            points = self._convert([vertex(*e) for e in [(-1, -1), (-1, 1), (1, 1), (1, -1)]])
        else:
            points = self._convert([vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]])

        polygon(self._screen, color, points)

        # # For debugging purposes
        # width = 0 if FILL_POLYGONS else 2
        # x1, x2 = points[0][0], points[2][0]
        # y1, y2 = points[0][1], points[2][1]
        # screen_x = x1 + (x2 - x1) / 2
        # screen_y = y1 + (y2 - y1) / 2
        # polygon(self._screen, color, points, width)
        # return screen_x, screen_y

    def _draw_arrow(self, pos, size, angle=None, cos=None, sin=None, color=(150, 150, 190)) -> None:
        if angle:
            cos, sin = np.cos(angle), np.sin(angle)
        self._rotated_box(pos,
                          size,
                          cos=(cos - sin) / np.sqrt(2),
                          sin=(cos + sin) / np.sqrt(2),
                          color=color,
                          centered=False)
        self._rotated_box(pos,
                          size,
                          cos=(cos + sin) / np.sqrt(2),
                          sin=(sin - cos) / np.sqrt(2),
                          color=color,
                          centered=False)

    def _draw_roads(self) -> None:
        # road_index_coordinates = [] # For debugging purposes
        for road in self._sim.roads:
            # Draw road background
            self._rotated_box(
                road.start,
                (road.length, 3.7),
                cos=road.angle_cos,
                sin=road.angle_sin,
                color=(180, 180, 220),
                centered=False
            )

            # # For debugging purposes
            # road_index_coordinates.append((road.index, screen_x, screen_y))

            # Draw road arrow
            if road.length > 5:
                for i in np.arange(-0.5 * road.length, 0.5 * road.length, 10):
                    pos = (road.start[0] + (road.length / 2 + i + 3) * road.angle_cos,
                           road.start[1] + (road.length / 2 + i + 3) * road.angle_sin)
                    self._draw_arrow(pos, (-1.25, 0.2), cos=road.angle_cos, sin=road.angle_sin)

        # # For debugging purposes
        # if DRAW_ROAD_IDS:
        #     # For debugging purposes
        #     for cords in road_index_coordinates:
        #         text_road_index = self._text_font.render(f'{cords[0]}', True, (0, 0, 0))
        #         self._screen.blit(text_road_index, (cords[1] - 5, cords[2] - 5))

    def _draw_vehicle(self, vehicle, road) -> None:
        l, h = vehicle.length, vehicle.width
        sin, cos = road.angle_sin, road.angle_cos
        x = road.start[0] + cos * vehicle.x
        y = road.start[1] + sin * vehicle.x
        self._rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)

        # # For debugging purposes
        # screen_x, screen_y = self._rotated_box((x, y), (l, h), cos=cos, sin=sin, centered=True)
        # if DRAW_VEHICLE_IDS:
        #     text_road_index = self._text_font.render(f'{vehicle.index}', True, (255, 255, 255),
        #                                              (0, 0, 0))
        #     self._screen.blit(text_road_index, (screen_x - 5, screen_y - 5))

    def _draw_vehicles(self) -> None:
        for i in self._sim.non_empty_roads:
            road = self._sim.roads[i]
            for vehicle in road.vehicles:
                self._draw_vehicle(vehicle, road)

    def _draw_signals(self) -> None:
        for signal in self._sim.traffic_signals:
            for i in range(len(signal.roads)):
                red, green = (255, 0, 0), (0, 255, 0)
                if signal.current_cycle == (False, False):
                    # Temp state, yellow color
                    yellow = (255, 255, 0)
                    color = yellow if signal.cycle[signal.current_cycle_index - 1][i] else red
                else:
                    color = green if signal.current_cycle[i] else red
                for road in signal.roads[i]:
                    a = 0
                    position = ((1 - a) * road.end[0] + a * road.start[0],
                                (1 - a) * road.end[1] + a * road.start[1])
                    self._rotated_box(position, (1, 3),
                                      cos=road.angle_cos, sin=road.angle_sin, color=color)

    def _draw_status(self):
        def render(text, color=(0, 0, 0), background=self._background_color):
            current_weather = self._sim.get_weather()
            #self.update_weather_visual(current_weather)
            return self._text_font.render(text, True, color, background)


        t = render(f'Time: {self._sim.t:.1f}')
        weather = render(f'Weather: {self._sim.current_weather}')  # Aggiungi la riga per il meteo
        if self._sim.max_gen:
            n_max_gen = render(f'Max Gen: {self._sim.max_gen}')
            self._screen.blit(n_max_gen, (10, 50))
        n_vehicles_generated = render(f'Vehicles Generated: {self._sim.n_vehicles_generated}')
        n_vehicles_on_map = render(f'Vehicles On Map: {self._sim.n_vehicles_on_map}')
        average_wait_time = render(f'Current Wait Time: {self._sim.current_average_wait_time:.1f}')
        self._screen.blit(t, (10, 20))
        self._screen.blit(weather, (10, 40))  # Mostra il meteo a una posizione definita
        self._screen.blit(n_vehicles_generated, (10, 70))
        self._screen.blit(n_vehicles_on_map, (10, 90))
        self._screen.blit(average_wait_time, (10, 120))

    def _draw_snow(self):
        """Disegna un effetto neve."""
        for _ in range(200):  # Numero di fiocchi
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 2)

    def _draw_weather_icon(self):
            """Disegna l'icona del meteo sulla finestra. Implementa il codice grafico effettivo qui."""
            print(f"Meteo corrente: {self.weather_visual}")  
        
    def update_weather_visual(self, weather: str):
        """Aggiorna la visualizzazione del meteo in base al tipo di meteo corrente."""
        if weather == "Sunny":
            self.weather_visual = "☀️ Soleggiato"  # Esempio testuale o usa icone grafiche reali
        elif weather == "Rain":
            self.weather_visual = "🌧️ Pioggia"
        elif weather == "Snow":
            self.weather_visual = "❄️ Neve"
        elif weather == "Fog":
            self.weather_visual = "🌫️ Nebbia"
        # Logica per disegnare graficamente il meteo nella finestra se hai un canvas o una GUI
        self._draw_weather_icon()
    
    def _draw_rain(self):
        """Disegna un effetto pioggia."""
        for _ in range(100):  # Numero di gocce
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            pygame.draw.line(self._screen, (0, 0, 255), (x, y), (x, y + 10), 2)
    
    def _draw(self):
        self._screen.fill(self._background_color)
        self._draw_roads()
        self._draw_vehicles()
        self._draw_signals()
        self._draw_status()

        if self._sim.current_weather == "Rain":
            self.rain_effect.update(self._screen)  
        elif self._sim.current_weather == "Snow":
            self.snow_effect.update(self._screen, self._sim.roads)  
        elif self._sim.current_weather == "Fog":
            self.fog_effect.update(self._screen)
        # elif self._sim.current_weather == "Sunny":
        #     self.sunny_effect.apply(self._screen)  

    

