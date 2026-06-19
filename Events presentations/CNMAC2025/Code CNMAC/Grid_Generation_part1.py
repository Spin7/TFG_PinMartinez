import osmnx as ox
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from shapely.geometry import Polygon, box
import time
import plotly.io as pio
from PIL import Image
import io
import requests
from io import BytesIO
import os
# Create directory for saving results
os.makedirs('figures', exist_ok=True)
os.makedirs('figures/rasters', exist_ok=True)

# Configure osmnx
ox.settings.timeout = 300
ox.settings.log_console = True

# Step 1: Define the area of interest
center_lat, center_lon = -25.2968215, -57.6533648
global gdf_edges 
def download_osm_data(center_lat, center_lon, attempt=1):
    """Download OSM data with multiple fallback strategies"""
    
    print("Trying graph_from_point as fallback...")
    distances = [400]  # meters
    
    for dist in distances:
        try:
            graph = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='all')
            gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
            print(f"Success with point query (dist={dist}m): {len(gdf_edges)} road segments")
            return gdf_nodes, gdf_edges
            
        except Exception as e:
            print(f"Point query with dist={dist}m failed: {e}")
            time.sleep(1)
    
    # If everything fails, return empty GeoDataFrames
    print("All download attempts failed. Using empty road data.")
    return gpd.GeoDataFrame(), gpd.GeoDataFrame()

# Download road data with retries
max_attempts = 3
gdf_nodes, gdf_edges = gpd.GeoDataFrame(), gpd.GeoDataFrame()

for attempt in range(1, max_attempts + 1):
    gdf_nodes, gdf_edges = download_osm_data(center_lat, center_lon, attempt)
    if not gdf_edges.empty:
        break
    time.sleep(2)  # Wait before retry

# Step 3: Categorize roads (if we have any)
if gdf_edges.empty:
    print("No road data available. Using empty categories.")
    categorized_roads = {
        'primary': gpd.GeoDataFrame(),
        'secondary': gpd.GeoDataFrame(),
        'tertiary': gpd.GeoDataFrame(),
        'residential': gpd.GeoDataFrame(),
        'other': gpd.GeoDataFrame()
    }
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
            # Get all other roads not in the specific categories
            exclude_types = ['primary', 'secondary', 'tertiary', 'residential']
            categorized_roads[category] = gdf_edges[~gdf_edges['highway'].isin(exclude_types)]
        else:
            categorized_roads[category] = gdf_edges[gdf_edges['highway'] == props['highway']]

# Function to save Mapbox view as raster image
def save_mapbox_view(center_lat, center_lon, zoom, style, save_path, width=2000, height=2000):
    """Save a Mapbox view as a raster image"""
    fig = go.Figure(go.Scattermapbox(
        lat=[center_lat],
        lon=[center_lon],
        mode='markers',
        marker=dict(size=1, color='red'),
        showlegend=False
    ))
    
    if style == "open-street-map":
        mapbox_config = dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        )
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
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        )
    
    fig.update_layout(
        mapbox=mapbox_config,
        margin=dict(l=0, r=0, t=0, b=0),
        width=width,
        height=height,
        paper_bgcolor='white'
    )
    
    try:
        # Save as high-quality PNG
        img_bytes = pio.to_image(fig, format='png', scale=2, engine='kaleido')
        img = Image.open(io.BytesIO(img_bytes))
        img.save(save_path)
        print(f"Saved {style} map to {save_path}")
        return np.array(img)
    except Exception as e:
        print(f"Error saving {style} map: {e}")
        return None

def create_roads_only_raster(roads_gdf, center_lat, center_lon, zoom, width, height, save_path):
    """Create a raster with only roads (no background)"""
    fig = go.Figure()
    
    # Add roads to the map with transparent background
    for category, roads in categorized_roads.items():
        if not roads.empty:
            for _, road in roads.iterrows():
                if hasattr(road.geometry, 'xy'):
                    lons, lats = road.geometry.xy
                    fig.add_trace(go.Scattermapbox(
                        lon=list(lons),
                        lat=list(lats),
                        mode='lines',
                        line=dict(width=road_categories[category]['width'], 
                                 color=road_categories[category]['color']),
                        name=category,
                        showlegend=False
                    ))
    
    fig.update_layout(
        mapbox=dict(
            style="white-bg",  # Transparent background
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    try:
        # Save as PNG with transparency
        img_bytes = pio.to_image(fig, format='png', scale=2, engine='kaleido')
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGBA to ensure transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        img.save(save_path)
        print(f"Saved roads-only raster to {save_path}")
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating roads-only raster: {e}")
        return None
# Save both map views as raster images
print("Saving OSM map as raster...")
osm_raster = save_mapbox_view(
    center_lat, center_lon, 
    zoom=17, 
    style="open-street-map",
    save_path="figures/rasters/osm_map.png",
    width=2000, 
    height=2000
)

print("Saving ESRI satellite map as raster...")
esri_raster = save_mapbox_view(
    center_lat, center_lon, 
    zoom=17, 
    style="satellite",
    save_path="figures/rasters/esri_satellite.png",
    width=2000, 
    height=2000
)
# Save road network as raster overlay with same dimensions
# Save roads-only raster (transparent background)
if not gdf_edges.empty:
    print("Creating roads-only raster (transparent background)...")
    roads_only_raster = create_roads_only_raster(
        gdf_edges, center_lat, center_lon,
        zoom=17,
        width=2000,
        height=2000,
        save_path="figures/rasters/roads_only.png"
    )