"""
===============================================================================
GRID GENERATOR FROM OSM GEOMETRIES (IMPROVED VERSION)

Inputs
------
CENTER_LAT
CENTER_LON
MAP_SIZE_M
CELL_SIZE_M

Outputs
-------
grid_data.npy
grid_visualization.png
environment_config.json

Additional Figures
------------------
figures/satellite_reference.png
figures/satellite_grid_overlay.png
figures/comparison_maps.png

Classes
-------
0 Empty
1 Road
2 Vegetation
3 Water
4 Building
5 Trap
===============================================================================
"""

import os
import json
import math
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, box
from matplotlib.colors import ListedColormap
import contextily as ctx


# =============================================================================
# USER PARAMETERS
# =============================================================================

CENTER_LAT = -25.2637
CENTER_LON = -57.5759

MAP_SIZE_M = 1000
CELL_SIZE_M = 3

TRAPS_CSV = "traps_coordinates_clean.csv"

os.makedirs("figures", exist_ok=True)


# =============================================================================
# CELL TYPES
# =============================================================================

CELL_TYPES = {
    0: {"name": "Empty Space",   "color": [242, 239, 233]},
    1: {"name": "Road",          "color": [100, 100, 100]},
    2: {"name": "Vegetation",    "color": [0, 150, 0]},
    3: {"name": "Water",         "color": [0, 0, 255]},
    4: {"name": "Building",      "color": [139, 69, 19]},
    5: {"name": "Trap",          "color": [255, 0, 255]},
}


# =============================================================================
# COORDINATE TRANSFORM
# =============================================================================

def latlon_to_local_meters(lat, lon, center_lat, center_lon):

    R = 6378137

    dlat = math.radians(lat - center_lat)
    dlon = math.radians(lon - center_lon)

    x = R * dlon * math.cos(math.radians(center_lat))
    y = R * dlat

    return x, y


# =============================================================================
# LOAD TRAPS
# =============================================================================

def load_traps(csv):

    df = pd.read_csv(csv)

    df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")

    df = df.dropna(subset=["Latitud", "Longitud"])

    print("\nLoaded traps:", len(df))

    return df


# =============================================================================
# DOWNLOAD OSM DATA
# =============================================================================

def download_osm(center_lat, center_lon, dist):

    print("Downloading buildings...")

    buildings = ox.features_from_point(
        (center_lat, center_lon),
        {"building": True},
        dist=dist
    )

    print("Downloading water...")

    water = ox.features_from_point(
        (center_lat, center_lon),
        {"natural": ["water", "wetland"]},
        dist=dist
    )

    print("Downloading vegetation...")

    vegetation = ox.features_from_point(
        (center_lat, center_lon),
        {
            "landuse": ["forest","grass","meadow","orchard","vineyard"],
            "leisure": ["park","garden"],
            "natural": ["wood","scrub"]
        },
        dist=dist
    )

    print("Downloading roads...")

    graph = ox.graph_from_point(
        (center_lat, center_lon),
        dist=dist,
        network_type="all"
    )

    nodes, roads = ox.graph_to_gdfs(graph)

    return buildings, water, vegetation, roads


# =============================================================================
# SATELLITE IMAGE
# =============================================================================

def download_satellite_image(center_lat, center_lon, map_size_m, filename):

    half = map_size_m / 2
    R = 6378137

    dlat = (half / R) * (180 / math.pi)
    dlon = (half / (R * math.cos(math.radians(center_lat)))) * (180 / math.pi)

    north = center_lat + dlat
    south = center_lat - dlat
    east = center_lon + dlon
    west = center_lon - dlon

    bbox = box(west, south, east, north)

    gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8,8))

    gdf.boundary.plot(ax=ax, linewidth=0)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    ax.set_axis_off()

    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    plt.close()

    print("Satellite image saved:", filename)


# =============================================================================
# GRID
# =============================================================================

class GridMap:

    def __init__(self, map_size_m, cell_size):

        self.map_size_m = map_size_m
        self.cell_size = cell_size

        self.size = int(map_size_m / cell_size)

        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)

        self.half = map_size_m / 2

        print("Grid:", self.size, "x", self.size)


    def cell_polygon(self, row, col, center_lat, center_lon):

        x = col * self.cell_size - self.half
        y = self.half - row * self.cell_size

        lat = center_lat + (y / 6378137) * (180 / math.pi)
        lon = center_lon + (x / 6378137) * (180 / math.pi) / math.cos(math.radians(center_lat))

        return Point(lon, lat).buffer(0.00001)


    def rasterize_polygons(self, gdf, class_id, center_lat, center_lon, min_cover):

        if gdf.empty:
            return

        print("Rasterizing class", class_id)

        gdf = gdf.to_crs("EPSG:4326")

        for row in range(self.size):
            for col in range(self.size):

                poly = self.cell_polygon(row, col, center_lat, center_lon)

                intersecting = gdf[gdf.intersects(poly)]

                if intersecting.empty:
                    continue

                overlap = 0

                for geom in intersecting.geometry:
                    overlap += geom.intersection(poly).area

                cover = overlap / poly.area

                if cover >= min_cover:

                    if class_id > self.grid[row, col]:
                        self.grid[row, col] = class_id


    def rasterize_roads(self, roads, center_lat, center_lon):

        if roads.empty:
            return

        print("Rasterizing roads")

        roads_buffer = roads.copy()

        roads_buffer["geometry"] = roads_buffer.buffer(0.00003)

        for row in range(self.size):
            for col in range(self.size):

                poly = self.cell_polygon(row, col, center_lat, center_lon)

                if roads_buffer.intersects(poly).any():

                    if self.grid[row, col] == 0:
                        self.grid[row, col] = 1


    def add_traps(self, df, center_lat, center_lon):

        trap_list = []

        for i, r in df.iterrows():

            lat = r["Latitud"]
            lon = r["Longitud"]

            x, y = latlon_to_local_meters(
                lat,
                lon,
                center_lat,
                center_lon
            )

            if abs(x) > self.half or abs(y) > self.half:
                continue

            col = int((x + self.half) / self.cell_size)
            row = int((self.half - y) / self.cell_size)

            if 0 <= row < self.size and 0 <= col < self.size:

                self.grid[row, col] = 5

                trap_list.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "x_m": float(x),
                    "y_m": float(y),
                    "row": int(row),
                    "col": int(col)
                })

        print("\nTraps inside map:", len(trap_list))

        return trap_list


    def visualize(self):

        ordered_keys = sorted(CELL_TYPES.keys())

        colors = [np.array(CELL_TYPES[k]["color"]) / 255 for k in ordered_keys]

        cmap = ListedColormap(colors)

        plt.figure(figsize=(8,8))

        plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(CELL_TYPES)-1)

        plt.title("Grid Map")

        plt.savefig("grid_visualization.png", dpi=300)

        plt.close()


# =============================================================================
# COMPARISON FIGURE
# =============================================================================

def comparison_figure():

    sat = plt.imread("figures/satellite_reference.png")
    grid = plt.imread("grid_visualization.png")

    fig, ax = plt.subplots(1,2, figsize=(12,6))

    ax[0].imshow(sat)
    #ax[0].set_title("Satellite")
    ax[0].axis("off")

    ax[1].imshow(grid)
    #ax[1].set_title("Generated Grid")
    ax[1].axis("off")

    plt.savefig("figures/comparison_maps.png", dpi=300)

    plt.close()

    print("Comparison figure saved.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\nCENTER:", CENTER_LAT, CENTER_LON)

    buildings, water, vegetation, roads = download_osm(
        CENTER_LAT,
        CENTER_LON,
        MAP_SIZE_M
    )

    download_satellite_image(
        CENTER_LAT,
        CENTER_LON,
        MAP_SIZE_M,
        "satellite_reference.png"
    )

    grid = GridMap(MAP_SIZE_M, CELL_SIZE_M)

    grid.rasterize_polygons(buildings, 4, CENTER_LAT, CENTER_LON, 0.30)
    grid.rasterize_polygons(water, 3, CENTER_LAT, CENTER_LON, 0.20)
    grid.rasterize_polygons(vegetation, 2, CENTER_LAT, CENTER_LON, 0.45)

    grid.rasterize_roads(roads, CENTER_LAT, CENTER_LON)

    traps = load_traps(TRAPS_CSV)

    trap_data = grid.add_traps(traps, CENTER_LAT, CENTER_LON)

    np.save("grid_data.npy", grid.grid)

    grid.visualize()

    comparison_figure()

    env = {
        "center_lat": CENTER_LAT,
        "center_lon": CENTER_LON,
        "map_size_m": MAP_SIZE_M,
        "cell_size_m": CELL_SIZE_M,
        "grid_size": grid.size,
        "traps": trap_data
    }

    with open("environment_params.json", "w") as f:
        json.dump(env, f, indent=4)

    print("\nFiles generated.")