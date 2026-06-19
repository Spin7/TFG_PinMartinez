import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.stats import mode
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import pandas as pd



def hazard_to_prob(rate, delta_t):
    # Probabilidad exacta de un proceso de Poisson (exponencial) en dt
    rate = max(rate, 0.0)
    return 1.0 - np.exp(-rate * delta_t)

# ==============================
#  ENTORNO
# ==============================
class Environment:
    def __init__(self, urban_grid_path, num_breeding_sites=15, K_per_site=100,
                 base_potential=100, lambda_decay=6):
        self.K_per_site = K_per_site
        self.urban_grid = self._downsample_grid(urban_grid_path)
        self.grid_height, self.grid_width = self.urban_grid.shape
        self.breeding_sites = self._place_breeding_sites(num_breeding_sites)
        self.breeding_capacity = {site: K_per_site for site in self.breeding_sites}

        # Distancias precomputadas (las puedes mantener si quieres)
        self.vegetation_distance = self._compute_distance(self.urban_grid == 2)
        self.water_distance = self._compute_distance(self.urban_grid == 3)
        self._update_breeding_distance()

        # Campo de potencial
        self.base_potential = base_potential
        self.lambda_decay = lambda_decay
        self._compute_potential_field()

        # -------------------
        # Variables climáticas
        # -------------------
        self.temperature = 25.0   # °C inicial
        self.humidity = 0.6       # 0-1 relativo
        self.weather_state = "clear"  # clear, cloudy, rain

    def _downsample_grid(self, urban_grid_path):
        """Carga y reduce la grilla original"""
        original_grid = np.load(urban_grid_path)
        original_grid[original_grid == 8] = 0
        original_grid[original_grid == 7] = 0
        original_grid[original_grid == 1] = 0

        downsampled = []
        for i in range(0, original_grid.shape[0], 2):
            row_blocks = []
            for j in range(0, original_grid.shape[1], 2):
                block = original_grid[i:i+10, j:j+10]
                if block.size > 0:
                    most_common = mode(block.flatten(), keepdims=False).mode
                    if isinstance(most_common, np.ndarray):
                        most_common = most_common.item()
                    row_blocks.append(most_common)
            if row_blocks:
                downsampled.append(row_blocks)
        return np.array(downsampled)


    def _place_breeding_sites(self, num_sites):
        # --- 1) base mask: permitido ---
        allowed = [0, 2, 8]  # empty, tree, empty space
        forbidden = [1, 3, 4, 5, 6, 7]  # roads, water, building
        
        suitable_mask = np.isin(self.urban_grid, allowed)
        suitable_sites = np.argwhere(suitable_mask)
        if len(suitable_sites) == 0:
            return []

        # --- 2) distancias ---
        water_mask = (self.urban_grid == 3)
        dist_to_water = distance_transform_edt(~water_mask)

        green_mask = (self.urban_grid == 2)
        dist_to_green = distance_transform_edt(~green_mask)

        road_mask = np.isin(self.urban_grid, [1, 5, 6, 7])
        dist_to_road = distance_transform_edt(~road_mask)

        # --- 3) pesos probabilísticos ---
        weights = []
        for (y, x) in suitable_sites:
            w = 1.0

            # favorecer cercanía al agua
            #w *= 1.0 / (1.0 + dist_to_water[y, x])

            # favorecer cercanía a árboles
            w *= 1.0 / (1.0 + dist_to_green[y, x])

            # penalizar cercanía a carreteras
            w *= (dist_to_road[y, x] + 1.0)

            weights.append(w)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()  # normalizar a probas

        # --- 4) muestreo ---
        num_sites = min(num_sites, len(suitable_sites))
        selected_indices = np.random.choice(
            len(suitable_sites), size=num_sites, replace=False, p=weights
        )
        return [tuple(suitable_sites[i]) for i in selected_indices]


    def _compute_distance(self, mask):
        return ndimage.distance_transform_edt(1 - mask.astype(float))

    def _update_breeding_distance(self):
        breeding_mask = np.zeros_like(self.urban_grid, dtype=float)
        for site in self.breeding_sites:
            breeding_mask[site] = 1
        self.breeding_distance = ndimage.distance_transform_edt(1 - breeding_mask)

    def _compute_potential_field(self, R=10):
        """Construye el campo de potencial a partir de criaderos y tipo de celda
        con corte a radio R de todos los sitios de cría.
        """
        potential = np.zeros((self.grid_height, self.grid_width), dtype=float)

        # Coordenadas de la grilla
        yy, xx = np.meshgrid(np.arange(self.grid_height),
                            np.arange(self.grid_width),
                            indexing="ij")

        # Inicializar distancia mínima a criaderos (para enmascarar luego)
        min_dist = np.full_like(yy, np.inf, dtype=float)

        # Sumar contribuciones de cada criadero
        for (sy, sx) in self.breeding_sites:
            dist = np.sqrt((yy - sy)**2 + (xx - sx)**2)
            contrib = self.base_potential * np.exp(-dist / self.lambda_decay)
            potential += contrib
            min_dist = np.minimum(min_dist, dist)  # actualizar mínima distancia

        # Enmascarar: fuera de R, peso = 0
        potential[min_dist > R] = 0.0

        # Ponderar por tipo de celda
        env_weights = {
            0: 1.0,   # libre
            2: 2.0,   # vegetación
            3: 0.0,   # agua (inaccesible)
            4: 0.5,   # construcciones
            5: 0.0,   # calles (repulsión)
            6: 0.0    # calles grandes (repulsión)
        }

        weighted_potential = np.zeros_like(potential)
        for cell_type, w in env_weights.items():
            mask = (self.urban_grid == cell_type)
            weighted_potential[mask] = potential[mask] * w

        # Guardamos el resultado
        self.potential_field = weighted_potential


    def update_breeding_sites(self, J, K_eff,
                          base_move=0.05,
                          base_expand=0.02,
                          base_extinct=0.01,
                          base_new=0.005):
        """
        Actualiza dinámicamente los sitios de cría según la presión poblacional.
        J: número actual de juveniles
        K_eff: capacidad de carga total actual (sumatoria de los sitios)
        """

        if K_eff <= 0:
            return

        # Fracción de saturación
        frac = J / K_eff

        new_sites = []

        for site in list(self.breeding_sites):
            y, x = site

            # --- Extinción: más probable si hay baja saturación (frac pequeño)
            p_extinct = base_extinct * max(0.0, 1.0 - frac)
            if np.random.rand() < p_extinct:
                self.breeding_sites.remove(site)
                continue

            # --- Movimiento: ruido ambiental (constante)
            if np.random.rand() < base_move:
                ny, nx = y + np.random.choice([-1, 0, 1]), x + np.random.choice([-1, 0, 1])
                if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                    self.urban_grid[ny, nx] not in [3, 5, 6]):
                    self.breeding_sites.remove(site)
                    self.breeding_sites.append((ny, nx))
                    site = (ny, nx)

            # --- Expansión: más probable si hay alta saturación (frac grande)
            p_expand = base_expand * frac
            if np.random.rand() < p_expand:
                dy, dx = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                new_y, new_x = y + dy, x + dx
                if (0 <= new_y < self.grid_height and 0 <= new_x < self.grid_width and
                    self.urban_grid[new_y, new_x] not in [3, 5, 6] and
                    (new_y, new_x) not in self.breeding_sites):
                    new_sites.append((new_y, new_x))

        # --- Generación aleatoria global: también depende de saturación
        p_new = base_new * frac
        if np.random.rand() < p_new:
            new_site = self.sample_random_position()
            if new_site not in self.breeding_sites:
                new_sites.append(new_site)

        # Actualizar lista
        self.breeding_sites.extend(new_sites)

        # Actualizar distancias y campo de potencial
        self._update_breeding_distance()
        self._compute_potential_field()

    # ==========================
    # CLIMA
    # ==========================
    def _update_temperature(self, t):
        """
        Actualiza la temperatura con una oscilación diurna simple.
        t: tiempo de simulación
        """
        base_temp = 25
        amp = 5
        self.temperature = base_temp + amp * np.sin(2 * np.pi * t)

    def _update_humidity(self, t):
        """
        Actualiza humedad: más alta en lluvia/noche, más baja en día despejado.
        """
        if self.weather_state == "rain":
            self.humidity = min(1.0, self.humidity + 0.05)
        elif self.weather_state == "clear":
            self.humidity = max(0.3, self.humidity - 0.02)
        elif self.weather_state == "cloudy":
            # tendencia neutra
            self.humidity = max(0.4, min(0.8, self.humidity + np.random.uniform(-0.01, 0.01)))

    def _update_weather_state(self, t):
        """
        Cambia entre clear, cloudy, rain con probabilidades dependientes de humedad.
        """
        rnd = np.random.rand()
        if self.weather_state == "clear":
            if rnd < 0.05:  # 5% chance pasar a nublado
                self.weather_state = "cloudy"
        elif self.weather_state == "cloudy":
            if rnd < 0.05:
                self.weather_state = "rain"
            elif rnd > 0.95:
                self.weather_state = "clear"
        elif self.weather_state == "rain":
            if rnd < 0.1:  # 10% chance dejar de llover
                self.weather_state = "cloudy"

    def update_climate(self, t):
        """Método maestro para actualizar clima."""
        self._update_weather_state(t)
        self._update_temperature(t)
        self._update_humidity(t)

    
    # ==========================
    # PARÁMETROS BIOLÓGICOS DEPENDIENTES DE LA TEMPERATURA
    # ==========================
    def get_temperature_dependent_params(self):
        T = self.temperature
        T_k = T + 273.15  # conversión a Kelvin

        # ---------------- Mortalidad juvenil μ1(T) ----------------
        # μ1(T) = 1 / [ c (T - T0)(Tm - T)^2 ]
        c_mu1, T0_mu1, Tm_mu1 = 0.0254, 2.3209, 31.0033
        if T <= T0_mu1 or T >= Tm_mu1:
            mu1 = float("inf")
        else:
            mu1 = 1.0 / (c_mu1 * (T - T0_mu1) * ((Tm_mu1 - T) ** 2))

        # ---------------- Mortalidad adulto μ2(T) ----------------
        # μ2(T) = 1 / [ c (T - T0)(Tm - T) ]
        c_mu2, T0_mu2, Tm_mu2 = 0.1037, 6.4429, 41.4382
        if T <= T0_mu2 or T >= Tm_mu2:
            mu2 = float("inf")
        else:
            mu2 = 1.0 / (c_mu2 * (T - T0_mu2) * (Tm_mu2 - T))

        # ---------------- Fecundidad b(T) ----------------
        # b(T) = c T (T - T0) sqrt(Tm - T)
        c_b, T0_b, Tm_b = 0.0058, 14.0343, 39.0899
        if T <= T0_b or T >= Tm_b:
            b = 0.0
        else:
            b = c_b * T * (T - T0_b) * np.sqrt(Tm_b - T)

        # ---------------- Desarrollo juvenil d(T) ----------------
        # d(T) = (a T_k e^(b (1/298.15 - 1/T_k))) / (298.15 (1 + e^(c (1/d - 1/T_k))))
        a_d, b_d, c_d, d_d = 0.1666, 21388, 32013, 300.02
        num = a_d * T_k * np.exp(b_d * (1 / 298.15 - 1 / T_k))
        den = 298.15 * (1 + np.exp(c_d * (1 / d_d - 1 / T_k)))
        d = num / den

        # Capacidad de carga K (puede ser constante o ligada a clima)
        K = 800

        return b, K, d, mu1, mu2

    # ------------------------------
    # Métodos de consulta para agentes
    # ------------------------------
    def get_possible_moves(self, pos, movement_range=14):
        """Devuelve lista de posiciones vecinas y su potencial"""
        y, x = pos
        height, width = self.urban_grid.shape

        y_min, y_max = max(0, y - movement_range), min(height, y + movement_range + 1)
        x_min, x_max = max(0, x - movement_range), min(width, x + movement_range + 1)

        candidates = []
        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if (ny, nx) == (y, x):
                    continue
                if self.urban_grid[ny, nx] in [3, 5, 6]:  # prohibidos
                    continue
                features = {
                    "cell_type": self.urban_grid[ny, nx],
                    "potential": self.potential_field[ny, nx],
                    "dist": np.sqrt((ny - y) ** 2 + (nx - x) ** 2),
                    "breeding_distance": self.breeding_distance[ny, nx],
                    "vegetation_distance": self.vegetation_distance[ny, nx],
                }
                candidates.append(((ny, nx), features))
        return candidates

    # def sample_random_position(self):
    #     """Devuelve una posición aleatoria ponderada por el potencial"""
    #     flat_potential = self.potential_field.flatten()
    #     flat_potential = np.maximum(flat_potential, 0)  # truncamos negativos
    #     if flat_potential.sum() == 0:
    #         # fallback: todos iguales
    #         flat_potential = np.ones_like(flat_potential)
    #     flat_potential /= flat_potential.sum()

    #     idx = np.random.choice(len(flat_potential), p=flat_potential)
    #     y, x = divmod(idx, self.grid_width)
    #     return (y, x)

    def sample_random_position(self):
        """Devuelve una posición completamente aleatoria, evitando agua y carreteras"""
        # Definir qué tipos de celdas están permitidos
        allowed = [0, 2, 4, 8]  # libre, vegetación, construcciones, vacío
        
        # Obtener todas las posiciones permitidas
        suitable_mask = np.isin(self.urban_grid, allowed)
        suitable_sites = np.argwhere(suitable_mask)

        if len(suitable_sites) == 0:
            # fallback: si no hay sitios válidos, elige cualquiera
            y = np.random.randint(0, self.grid_height)
            x = np.random.randint(0, self.grid_width)
            return (y, x)

        # Elegir al azar entre los sitios válidos
        idx = np.random.randint(len(suitable_sites))
        return tuple(suitable_sites[idx])
    
    def nearest_breeding_site(self, pos):
        if not self.breeding_sites:
            return None
        py, px = pos
        dists = [np.hypot(py - y, px - x) for (y, x) in self.breeding_sites]
        return self.breeding_sites[int(np.argmin(dists))]


# ==============================
#  AGENTES
# ==============================
class Agent:
    def __init__(self, agent_id, pos):
        self.id = agent_id
        self.pos = pos

    def dies(self):
        pass


class Juvenile(Agent):
    def transition(self, d, mu1, delta_t):
        """
        Devuelve: 'mature', 'die' o None (sigue como juvenil)
        Implementa riesgos competidores con tasas d y mu1.
        """
        r_total = d + mu1
        if r_total <= 0:
            return None
        p_event = hazard_to_prob(r_total, delta_t)
        if random.random() >= p_event:
            return None
        # ocurrió un evento: decidir cuál
        if random.random() < (d / r_total):
            return "mature"
        else:
            return "die"


class Adult(Agent):
    def __init__(self, agent_id, pos, bias_breeding=1.0, bias_vegetation=1.0, bias_random=10):
        super().__init__(agent_id, pos)
        """
        bias_breeding   → cuánto le atraen los criaderos
        bias_vegetation → cuánto le atrae la vegetación
        bias_random     → peso de exploración aleatoria
        """
        self.bias_breeding = bias_breeding
        self.bias_vegetation = bias_vegetation
        self.bias_random = bias_random

    def move(self, env: Environment, movement_range=14):
        """El adulto decide movimiento usando potencial del entorno + sesgos propios."""
        candidates = env.get_possible_moves(self.pos, movement_range)

        moves, weights = [], []
        for (ny, nx), f in candidates:
            # Partimos del potencial base del entorno
            w = f["potential"]

            # Sesgos individuales
            # Atracción hacia criaderos
            #w *= np.exp(-f["breeding_distance"] / 15.0) * self.bias_breeding

            # Atracción hacia vegetación
            #w *= np.exp(-f["vegetation_distance"] / 12.0) * self.bias_vegetation

            # Exploración aleatoria
            w += self.bias_random

            # Penalización por distancia
            w /= (f["dist"] + 1)

            moves.append((ny, nx))
            weights.append(max(w, 1e-6))  # evitar ceros exactos

        if moves:
            weights = np.array(weights, dtype=float)
            weights /= np.sum(weights)
            self.pos = moves[np.random.choice(len(moves), p=weights)]
        
    def dies(self, mu, delta_t):
        return random.random() < hazard_to_prob(mu, delta_t)

# ==============================
#  SIMULACIÓN
# ==============================
class MosquitoSimulation:
    def __init__(self, env: Environment, params, delta_t):
        self.env = env
        #self.b, self.k, self.d, self.mu1, self.mu2 = params
        self.b, self.k, self.d, self.mu1, self.mu2 = self.env.get_temperature_dependent_params()
        self.juveniles = {}
        self.adults = {}
        self.time = 0
        self.next_id = 0
        self.delta_t = delta_t

        self.time_hist = []  

        # tracking poblacional
        self.total_population = []
        self.adult_population = []
        self.juvenile_population = []

        # tracking climático y biológico
        self.temperature_hist = []
        self.humidity_hist = []
        self.weather_hist = []
        self.b_hist = []
        self.d_hist = []
        self.mu1_hist = []
        self.mu2_hist = []

        # tracking juveniles por sitio de cria
        self.juveniles_per_site_hist = {site: [] for site in self.env.breeding_sites}

    def add_juvenile(self, site):
        j = Juvenile(self.next_id, site)
        self.juveniles[self.next_id] = j
        self.next_id += 1

    def add_adult(self, pos):
        a = Adult(self.next_id, pos)
        self.adults[self.next_id] = a
        self.next_id += 1

    def initialize_population(self, J0, A0):
        for _ in range(J0):
            site = random.choice(self.env.breeding_sites)
            self.add_juvenile(site)
        for _ in range(A0):
            pos = self.env.sample_random_position()
            self.add_adult(pos)

    def step(self):
        self.time += self.delta_t

        # --- snapshot poblacional para las tasas de este paso ---
        J = len(self.juveniles)
        A = len(self.adults)

        # Usa K del modelo ODE (param self.k); si prefieres capacidad espacial: 
        # K_eff = sum(self.env.breeding_capacity.values())
        K_eff = self.k

        # === ADULTOS: calcular acciones ===
        adults_list = list(self.adults.values())
        random.shuffle(adults_list)

        survivors = {}
        deaths = set()
        moves = {}

        for a in adults_list:
            a.move(self.env)              # mueve al adulto (actualiza su .pos)
            if a.dies(self.mu2, self.delta_t):
                deaths.add(a.id)
            else:
                moves[a.id] = a.pos       # posición ya actualizada
                survivors[a.id] = a

        # === JUVENILES: calcular transiciones ===
        juveniles_list = list(self.juveniles.values())
        random.shuffle(juveniles_list)

        j_survivors = {}
        j_deaths = set()
        j_to_adults = []

        for j in juveniles_list:
            outcome = j.transition(self.d, self.mu1, self.delta_t)
            if outcome == "die":
                j_deaths.add(j.id)
            elif outcome == "mature":
                if random.random() < 0.5:
                    j_to_adults.append(j.pos)
                j_deaths.add(j.id)
            else:
                j_survivors[j.id] = j

        # === NACIMIENTOS: calcular huevos ===
        juveniles_por_sitio = {s: 0 for s in self.env.breeding_sites}
        for j in j_survivors.values():
            juveniles_por_sitio[j.pos] += 1

        tau_rep = 10.0
        new_juveniles = []

        adults_for_reproduction = list(survivors.values())
        random.shuffle(adults_for_reproduction)

        for a in adults_for_reproduction:
            affinities = {}
            for s in self.env.breeding_sites:
                d = np.linalg.norm(np.array(a.pos) - np.array(s))
                affinities[s] = np.exp(-d / tau_rep)

            total_aff = sum(affinities.values())
            if total_aff <= 0:
                continue

            probs = {s: affinities[s] / total_aff for s in self.env.breeding_sites}

            lam_sites = {}
            for s, p_s in probs.items():
                J_s = juveniles_por_sitio[s]
                K_s = self.env.K_per_site
                psi_s = max(0.0, 1.0 - J_s / max(K_s, 1e-9))
                lam_sites[s] = self.b * p_s * psi_s * self.delta_t

            for s, lam_dt in lam_sites.items():
                if lam_dt > 0:
                    n_offspring = np.random.poisson(lam_dt)
                    for _ in range(n_offspring):
                        new_juveniles.append(s)
                    juveniles_por_sitio[s] += n_offspring

        # === ETAPA FINAL: aplicar cambios sincrónicamente ===

        # actualizar adultos
        self.adults = {}
        for a_id, a in survivors.items():
            a.pos = moves[a_id]   # aplicar movimiento
            self.adults[a_id] = a
        for pos in j_to_adults:   # agregar los que maduraron
            self.add_adult(pos)

        # actualizar juveniles
        self.juveniles = {j_id: j for j_id, j in j_survivors.items()}
        for pos in new_juveniles:
            self.add_juvenile(pos)


        # Actualizar clima y parámetros dependientes
        self.env.update_climate(self.time)
        self.b, self.k, self.d, self.mu1, self.mu2 = self.env.get_temperature_dependent_params()

        self.update_tracking()



    def update_tracking(self):
        self.time_hist.append(self.time)
        # Guardar historial de la poblacion
        self.total_population.append(len(self.adults) + len(self.juveniles))
        self.adult_population.append(len(self.adults))
        self.juvenile_population.append(len(self.juveniles))

        # Guardar historial de parametros climático y biológico
        self.temperature_hist.append(self.env.temperature)
        self.humidity_hist.append(self.env.humidity)
        self.weather_hist.append(self.env.weather_state)
        self.b_hist.append(self.b)
        self.d_hist.append(self.d)
        self.mu1_hist.append(self.mu1)
        self.mu2_hist.append(self.mu2)

        # Guardar historial de juveniles por sitio
        count_per_site = {site: 0 for site in self.env.breeding_sites}
        for j in self.juveniles.values():
            if j.pos in count_per_site:
                count_per_site[j.pos] += 1
        for site in self.env.breeding_sites:
            self.juveniles_per_site_hist[site].append(count_per_site.get(site, 0))


    def visualize_map_simulation_at_step(self, step):
        # ====== MAPA ESPACIAL ======
        plt.figure(figsize=(7, 6))
        cmap = ListedColormap([
                [0.9, 0.9, 0.9], [0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0], [0.0, 0.0, 0.8],
                [0.5, 0.3, 0.1], [93/255, 98/255, 99/255],
                [0.9, 0.6, 0.6], [0.0, 0.0, 0.0]
            ])
        plt.imshow(self.env.urban_grid, cmap=cmap, vmin=0, vmax=7)

        if self.env.breeding_sites:
            y, x = zip(*self.env.breeding_sites)
            plt.scatter(x, y, c="purple", s=30, marker="s", edgecolors="black")

        if self.adults:
            ay, ax = zip(*[a.pos for a in self.adults.values()])
            plt.scatter(ax, ay, c="black", s=1, label="Adults")

        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(f"map_at_day_{int(step*self.delta_t)}.png", dpi=300, bbox_inches="tight", pad_inches=0)

        plt.close()

    def get_map_state_at_step(self):
        bs_y, bs_x = zip(*self.env.breeding_sites)
        a_y, a_x = zip(*[a.pos for a in self.adults.values()])
        return bs_y, bs_x, a_y, a_x



    def create_history_plot_at_step(self, step):
        t = self.time_hist
        # FIGURA 1: dinámica poblacional
        plt.figure(figsize=(7, 6))
        plt.plot(t, self.total_population, "k-", label="Total")
        plt.plot(t, self.adult_population, "r-", label="Adults")
        plt.plot(t, self.juvenile_population, "b-", label="Juveniles")
        plt.legend()
        plt.title("Population variation over time")
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.savefig(f"population_dynamics_step_{step}.png", dpi=300,
                        bbox_inches="tight")
        plt.close()

        # FIGURA 2: juveniles por sitio
        plt.figure(figsize=(7, 6))
        for site, hist in self.juveniles_per_site_hist.items():
            plt.plot(t, hist, alpha=0.7,
                    label=f"Site {(float(site[0]), float(site[1]))}")
        plt.legend(ncol=2, fontsize=8)
        plt.title("Juveniles per breeding site")
        plt.xlabel("Days")
        plt.ylabel("Juveniles")
        plt.savefig(f"juveniles_per_site_step_{step}.png", dpi=300,
                        bbox_inches="tight")
        plt.close()

    def get_history_tracking(self):
        return self.time_hist, self.total_population, self.adult_population, self.juvenile_population, self.juveniles_per_site_hist


    def show_grid(self, filename="grid_map.png"):
            fig = plt.figure(figsize=(14, 10), dpi=100)
            cmap = ListedColormap([
                [0.9, 0.9, 0.9], [0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0], [0.0, 0.0, 0.8],
                [0.5, 0.3, 0.1], [93/255, 98/255, 99/255],
                [0.9, 0.6, 0.6], [0.0, 0.0, 0.0]
            ])

            plt.imshow(self.env.urban_grid, cmap=cmap, vmin=0, vmax=7)
            plt.axis("off")
            plt.subplots_adjust(0, 0, 1, 1)  # quita bordes y márgenes
            
            plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)


    def run(self, t_max, visualize_intervals_steps):
            for step in range(int(t_max / self.delta_t)):
                self.step()
                if step in visualize_intervals_steps:
                    print(f"Step {step}: Adults={len(self.adults)}, Juveniles={len(self.juveniles)}")
                    self.visualize_map_simulation_at_step(step)
            self.step()
            step = step +1
            self.visualize_map_simulation_at_step(step)
            self.create_history_plot_at_step(step)


    def run_and_get_tracking(self, t_max):
        for step in range(int(t_max / self.delta_t)):
            self.step()
        self.step()
        return self.get_history_tracking()
    




# ==============================
#  USO
# ==============================
if __name__ == "__main__":
    params = (3, 800, 0.04, 1/25, 1/25)  # b, k, d, mu1, mu2
    env = Environment("grid_data.npy", num_breeding_sites=15, K_per_site=100)

    def run_multiple_times(simulation_runs, t_max=500, J0=60, A0=40):
        print(f"Empezando a Correr {simulation_runs} simulaciones")
        for run in range(simulation_runs):
            sim = MosquitoSimulation(env, params, delta_t=0.1)
            sim.initialize_population(J0=J0, A0=A0)
            t, total_hist, adult_hist, juvenile_hist, juvenile_per_site_hist = sim.run_and_get_tracking(t_max=t_max)

            # Convertimos a DataFrame para guardar
            df = pd.DataFrame({
                "time": t,
                "total": total_hist,
                "adults": adult_hist,
                "juveniles": juvenile_hist,
            })

            # Agregar columnas de juveniles por sitio
            for site, hist in juvenile_per_site_hist.items():
                site_name = f"site_{site[0]}_{site[1]}"
                df[site_name] = hist

            # Guardar resultados en Excel (una hoja por simulación)
            filename = f"simulacion_{run+1}.xlsx"
            df.to_excel(filename, index=False)

            print(f"✅ Resultados guardados en {filename}")

        print("🎉 Todas las simulaciones finalizadas y guardadas")

    # Ejecutamos
    run_multiple_times(simulation_runs=100, t_max=500, J0=60, A0=40)