import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2
import time
import os
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.io as pio

# Crear directorios
os.makedirs('figures', exist_ok=True)
os.makedirs('figures/rasters', exist_ok=True)

# Configuración osmnx
ox.settings.timeout = 300
ox.settings.log_console = True

# -----------------------
# FUNCIONES OSM
# -----------------------
center_lat, center_lon = -25.2968215, -57.6533648
global gdf_edges 

def download_osm_data(center_lat, center_lon, attempt=1):
    print("Trying graph_from_point as fallback...")
    distances = [400]  # metros
    for dist in distances:
        try:
            graph = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='all')
            gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
            print(f"Success with point query (dist={dist}m): {len(gdf_edges)} road segments")
            return gdf_nodes, gdf_edges
        except Exception as e:
            print(f"Point query with dist={dist}m failed: {e}")
            time.sleep(1)
    print("All download attempts failed. Using empty road data.")
    return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# Descargar red vial
max_attempts = 3
gdf_nodes, gdf_edges = gpd.GeoDataFrame(), gpd.GeoDataFrame()
for attempt in range(1, max_attempts + 1):
    gdf_nodes, gdf_edges = download_osm_data(center_lat, center_lon, attempt)
    if not gdf_edges.empty:
        break
    time.sleep(2)

# Categorizar caminos
if gdf_edges.empty:
    categorized_roads = {k: gpd.GeoDataFrame() for k in ['primary','secondary','tertiary','residential','other']}
else:
    road_categories = {
        'primary': {'highway': 'primary', 'color': '#ff0000', 'width':14 },
        'secondary': {'highway': 'secondary', 'color': '#ff0000', 'width': 12},
        'tertiary': {'highway': 'tertiary', 'color':'#ff0000', 'width': 12},
        'residential': {'highway': 'residential', 'color': '#ff0000', 'width': 10},
        'other': {'highway': True, 'color': '#dddddd', 'width': 14}
    }
    categorized_roads = {}
    for category, props in road_categories.items():
        if category == 'other':
            exclude_types = ['primary','secondary','tertiary','residential']
            categorized_roads[category] = gdf_edges[~gdf_edges['highway'].isin(exclude_types)]
        else:
            categorized_roads[category] = gdf_edges[gdf_edges['highway'] == props['highway']]

# -----------------------
# FUNCIONES DE MAPBOX
# -----------------------
def save_mapbox_view(center_lat, center_lon, zoom, style, save_path, width=2000, height=2000):
    fig = go.Figure(go.Scattermapbox(
        lat=[center_lat], lon=[center_lon],
        mode='markers', marker=dict(size=1, color='red'), showlegend=False
    ))
    if style == "open-street-map":
        mapbox_config = dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom)
    else:
        mapbox_config = dict(
            style="white-bg",
            layers=[{
                'below': 'traces',
                'sourcetype': 'raster',
                'source': [
                    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ]
            }],
            center=dict(lat=center_lat, lon=center_lon), zoom=zoom
        )
    fig.update_layout(
        mapbox=mapbox_config, margin=dict(l=0,r=0,t=0,b=0),
        width=width, height=height, paper_bgcolor='white'
    )
    img_bytes = pio.to_image(fig, format='png', scale=2, engine='kaleido')
    img = Image.open(io.BytesIO(img_bytes))
    img.save(save_path)
    return np.array(img)

def create_roads_only_raster(roads_gdf, center_lat, center_lon, zoom, width, height, save_path):
    fig = go.Figure()
    for category, roads in categorized_roads.items():
        if not roads.empty:
            for _, road in roads.iterrows():
                if hasattr(road.geometry, 'xy'):
                    lons, lats = road.geometry.xy
                    fig.add_trace(go.Scattermapbox(
                        lon=list(lons), lat=list(lats),
                        mode='lines',
                        line=dict(width=road_categories[category]['width'],
                                  color=road_categories[category]['color']),
                        showlegend=False
                    ))
    fig.update_layout(
        mapbox=dict(style="white-bg", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        margin=dict(l=0,r=0,t=0,b=0),
        width=width, height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    img_bytes = pio.to_image(fig, format='png', scale=2, engine='kaleido')
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGBA': img = img.convert('RGBA')
    img.save(save_path)
    return np.array(img)

# -----------------------
# CLASE GRID MAP CON CELDAS EN METROS
# -----------------------
class Grid_Map:
    def __init__(self, map_size_m, cell_size_m):
        self.map_size_m = map_size_m
        self.cell_size_m = cell_size_m
        self.width = int(map_size_m // cell_size_m)
        self.height = int(map_size_m // cell_size_m)
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.cell_types = {
            0: {'name': 'Empty Space', 'color': [242, 239, 233]},
            1: {'name': 'Road', 'color': [100, 100, 100]},
            2: {'name': 'Tree', 'color': [0, 150, 0]},
            3: {'name': 'Water', 'color': [0, 0, 255]},
            4: {'name': 'Building', 'color': [139, 69, 19]},
        }

    def load_and_process_rasters(self, roads_path, satellite_path, osm_path):
        print("Loading raster images...")
        roads_img = Image.open(roads_path).convert('RGB')
        satellite_img = Image.open(satellite_path).convert('RGB')
        osm_img = Image.open(osm_path).convert('RGB')

        # Redimensionar a dimensiones de la grilla
        roads_img = roads_img.resize((self.width, self.height))
        satellite_img = satellite_img.resize((self.width, self.height))
        osm_img = osm_img.resize((self.width, self.height))

        roads_array = np.array(roads_img)
        satellite_array = np.array(satellite_img)
        osm_array = np.array(osm_img)

        self._process_osm_features(osm_array)
        self._process_empty_spaces(osm_array)  
        self._process_vegetation(satellite_array)
        self._process_roads(roads_array)
        print("Grid processing complete!")

    def _process_empty_spaces(self, osm_array):
        empty_color = np.array([242, 239, 233])
        color_tolerance = 10  
        empty_mask = np.all(np.abs(osm_array - empty_color) <= color_tolerance, axis=2)
        self.grid[empty_mask] = 0

    def _process_roads(self, roads_array):
        roads_gray = np.mean(roads_array, axis=2)
        road_mask = roads_gray < 220
        self.grid[road_mask] = 1

    def _process_vegetation(self, satellite_array):
        hsv = cv2.cvtColor(satellite_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        self.grid[green_mask > 0] = 2

    def _process_osm_features(self, osm_array):
        hsv = cv2.cvtColor(osm_array, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        building_mask = self._detect_buildings(osm_array, hsv)
        self.grid[blue_mask > 0] = 3
        self.grid[building_mask > 0] = 4

    def _detect_buildings(self, rgb_array, hsv_array):
        target_color = np.array([217, 208, 201])
        color_tolerance = 40
        color_mask = np.all(np.abs(rgb_array - target_color) < color_tolerance, axis=2)
        kernel = np.ones((3, 3), np.uint8)
        building_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return building_mask > 0

    def create_visualization(self, save_path='cellular_automaton_grid.png'):
        stats = self.get_grid_statistics(return_dict=True)
        colors = [np.array(self.cell_types[i]['color'])/255.0 for i in range(len(self.cell_types))]
        cmap = ListedColormap(colors)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        im = ax1.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(self.cell_types)-1)
        ax1.set_title(f'Grid Map ({self.cell_size_m} m por celda)', fontsize=16)
        ax1.set_xlabel('X (celdas)')
        ax1.set_ylabel('Y (celdas)')
        legend_patches = []
        for cell_id, cell_info in self.cell_types.items():
            percentage = stats[cell_info['name']]['percentage']
            label = f"{cell_info['name']} ({percentage:.1f}%)"
            patch = mpatches.Patch(color=np.array(cell_info['color'])/255.0, label=label)
            legend_patches.append(patch)
        ax2.legend(handles=legend_patches, loc='center', fontsize=12)
        ax2.axis('off')
        ax2.set_title('Legend (with %)', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved to {save_path}")

    def get_grid_statistics(self, return_dict=False):
        total_cells = self.width * self.height
        stats = {}
        for cell_id, cell_info in self.cell_types.items():
            count = np.sum(self.grid == cell_id)
            percentage = (count / total_cells) * 100
            stats[cell_info['name']] = {'count': count, 'percentage': percentage}
        print("\n=== GRID STATISTICS ===")
        for feature, data in stats.items():
            print(f"{feature}: {data['count']} cells ({data['percentage']:.1f}%)")
        if return_dict:
            return stats
        return None

    def save_grid_data(self, save_path='grid_data.npy'):
        np.save(save_path, self.grid)
        print(f"Grid data saved to {save_path}")

# -----------------------
# FUNCIONES DE CÁLCULO EXTENSIÓN REAL
# -----------------------
def estimate_map_extent(lat, zoom, img_size_px):
    """
    Calcula la extensión del mapa en metros para un centro dado,
    un nivel de zoom y tamaño de imagen en píxeles (lado).
    """
    earth_circumference = 40075016.686  # metros
    meters_per_pixel = (earth_circumference * np.cos(np.radians(lat))) / (2**(zoom+8))
    extent_m = meters_per_pixel * img_size_px
    return extent_m, meters_per_pixel

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # Config inicial
    zoom = 17
    img_size_px = 1400

    # Calcular extensión real del mapa
    extent_m, m_per_px = estimate_map_extent(center_lat, zoom, img_size_px)
    print(f"\n=== MAP EXTENT ESTIMATE ===")
    print(f"Zoom: {zoom}")
    print(f"Resolución: {m_per_px:.2f} m/pixel")
    print(f"Extensión total: {extent_m:.1f} m × {extent_m:.1f} m\n")

    # Guardar rasters base
    print("Saving OSM map as raster...")
    osm_raster = save_mapbox_view(center_lat, center_lon, zoom, "open-street-map", "figures/rasters/osm_map.png", img_size_px, img_size_px)
    print("Saving ESRI satellite map as raster...")
    esri_raster = save_mapbox_view(center_lat, center_lon, zoom, "satellite", "figures/rasters/esri_satellite.png", img_size_px, img_size_px)

    if not gdf_edges.empty:
        print("Creating roads-only raster...")
        roads_only_raster = create_roads_only_raster(gdf_edges, center_lat, center_lon, zoom, img_size_px, img_size_px, "figures/rasters/roads_only.png")

    # ============================
    #  AQUÍ SE DEFINE EL ÁREA A USAR
    # ============================
    # Por defecto usa la extensión calculada, pero se puede ajustar
    user_map_size_m = 1000  # <<--- CAMBIA ESTE VALOR (ej: 1000, 1500, 3000, etc.)
    print(f"Usando extensión definida por el usuario: {user_map_size_m} m × {user_map_size_m} m")

    # Crear grilla con área ajustada
    grid = Grid_Map(map_size_m=user_map_size_m, cell_size_m=1)

    roads_path = "figures/rasters/roads_only.png"
    satellite_path = "figures/rasters/esri_satellite.png"
    osm_path = "figures/rasters/osm_map.png"

    grid.load_and_process_rasters(roads_path, satellite_path, osm_path)
    grid.save_grid_data()
    grid.create_visualization('cellular_automaton_grid.png')
    grid.get_grid_statistics()
