import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2

class CellularAutomatonGrid:
    def __init__(self, width=1400, height=1000):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.cell_types = {
            0: {'name': 'Empty', 'color': [200, 200, 200]},
            1: {'name': 'Road', 'color': [100, 100, 100]},
            2: {'name': 'Tree', 'color': [0, 150, 0]},
            3: {'name': 'Water', 'color': [0, 0, 255]},
            4: {'name': 'Building', 'color': [139, 69, 19]},
            5: {'name': 'Primary Road', 'color': [93, 98, 99]},
            6: {'name': 'Secondary Road', 'color': [93, 98, 99]},
            7: {'name': 'Residential Road', 'color': [93, 98, 99]},
            8: {'name': 'Empty Space', 'color': [242, 239, 233]}  # NEW: Empty space category
        }
    
    def load_and_process_rasters(self, roads_path, satellite_path, osm_path):
        """Load and process the three raster images"""
        print("Loading raster images...")
        
        # Load images
        roads_img = Image.open(roads_path).convert('RGB')
        satellite_img = Image.open(satellite_path).convert('RGB')
        osm_img = Image.open(osm_path).convert('RGB')
        
        # Resize to grid dimensions
        roads_img = roads_img.resize((self.width, self.height))
        satellite_img = satellite_img.resize((self.width, self.height))
        osm_img = osm_img.resize((self.width, self.height))
        
        # Convert to numpy arrays
        roads_array = np.array(roads_img)
        satellite_array = np.array(satellite_img)
        osm_array = np.array(osm_img)
        
        
        
        print("Processing water and buildings from OSM...")
        self._process_osm_features(osm_array)
        
        print("Processing empty spaces from OSM...")
        self._process_empty_spaces(osm_array)  # Process empty spaces FIRST
        print("Processing vegetation from satellite...")
        self._process_vegetation(satellite_array)
        
        print("Processing road information...")
        self._process_roads(roads_array)
        
        print("Grid processing complete!")
    
    def _process_empty_spaces(self, osm_array):
        """Detect and mark empty spaces (color: 242, 239, 233)"""
        # Target empty space color
        empty_color = np.array([242, 239, 233])
        color_tolerance = 10  # Allow some variation
        
        # Create mask for empty spaces
        empty_mask = np.all(np.abs(osm_array - empty_color) <= color_tolerance, axis=2)
        
        # Mark empty spaces (category 8)
        self.grid[empty_mask] = 8
    
    def _process_roads(self, roads_array):
        """Extract road information with categories"""
        # Convert to grayscale for road detection
        roads_gray = np.mean(roads_array, axis=2)
        
        # Detect roads (darker areas)
        road_mask = roads_gray < 220
        
        # Categorize roads by color (simplified)
        for y in range(self.height):
            for x in range(self.width):
                if road_mask[y, x] and self.grid[y, x] != 8:  # Don't overwrite empty spaces
                    r, g, b = roads_array[y, x]
                    
                    # Primary roads - pure red (255,0,0) or very close
                    if r > 240 and g < 30 and b < 30:
                        self.grid[y, x] = 5  # Primary road
                  
                    # Secondary roads - light pink/red
                    elif r >= 220 and 210 >= g >= 180 and 210 >= b >= 180:
                        self.grid[y, x] = 6  # Secondary road
                        
        
    def _process_vegetation(self, satellite_array):
        """Extract vegetation from satellite imagery"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(satellite_array, cv2.COLOR_RGB2HSV)
        
        # Green color range for vegetation
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply vegetation where detected (and not already occupied by roads or empty spaces)
        vegetation_mask = green_mask > 0
        for y in range(self.height):
            for x in range(self.width):
                if vegetation_mask[y, x]:  # Can replace empty or general empty
                    self.grid[y, x] = 2
    
    def _process_osm_features(self, osm_array):
        """Extract water and buildings from OSM map"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(osm_array, cv2.COLOR_RGB2HSV)
        
        # Blue color range for water
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Improved building detection using multiple approaches
        building_mask = self._detect_buildings(osm_array, hsv)
        
        # Apply features (prioritizing existing features)
        for y in range(self.height):
            for x in range(self.width):
                # Water (blue areas) - can replace vegetation but not roads or empty spaces
                if blue_mask[y, x] and self.grid[y, x] in [0, 2, 8]:  # Can replace vegetation or empty
                    self.grid[y, x] = 3
                # Buildings - can replace empty spaces but not roads
                elif building_mask[y, x] and self.grid[y, x] in [0, 8]:  # Can replace empty
                    self.grid[y, x] = 4
    
    def _detect_buildings(self, rgb_array, hsv_array):
        """Detect buildings using multiple color detection methods"""
        
        # Method 4: Detect specific RGB color (217, 208, 201) with tolerance
        target_color = np.array([217, 208, 201])
        color_tolerance = 40
        color_mask = np.all(np.abs(rgb_array - target_color) < color_tolerance, axis=2)
        
        # Combine all building detection methods
        building_mask =  color_mask
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        building_mask = cv2.morphologyEx(building_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        return building_mask > 0
    
    def create_visualization(self, save_path='cellular_automaton_grid.png'):
        """Create a visualization of the grid with legend"""
        # Create color map
        colors = []
        for i in range(len(self.cell_types)):
            colors.append(np.array(self.cell_types[i]['color']) / 255.0)
        cmap = ListedColormap(colors)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot grid
        im = ax1.imshow(self.grid, cmap=cmap, vmin=0, vmax=len(self.cell_types)-1)
        ax1.set_title('Cellular Automaton Grid', fontsize=16)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # Create legend
        legend_patches = []
        for cell_type in self.cell_types.values():
            patch = mpatches.Patch(color=np.array(cell_type['color']) / 255.0, 
                                  label=cell_type['name'])
            legend_patches.append(patch)
        
        ax2.legend(handles=legend_patches, loc='center', fontsize=12)
        ax2.axis('off')
        ax2.set_title('Legend', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved to {save_path}")
    
    def get_grid_statistics(self):
        """Print statistics about the grid"""
        total_cells = self.width * self.height
        stats = {}
        
        for cell_id, cell_info in self.cell_types.items():
            count = np.sum(self.grid == cell_id)
            percentage = (count / total_cells) * 100
            stats[cell_info['name']] = {'count': count, 'percentage': percentage}
        
        print("\n=== GRID STATISTICS ===")
        for feature, data in stats.items():
            print(f"{feature}: {data['count']} cells ({data['percentage']:.1f}%)")
        
        return stats
    
    def save_grid_data(self, save_path='grid_data.npy'):
        """Save grid data for later use"""
        np.save(save_path, self.grid)
        print(f"Grid data saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Initialize grid
    print("Initializing cellular automaton grid...")
    grid = CellularAutomatonGrid(width=1400, height=1000)
    
    # Paths to your raster files
    roads_path = "figures/rasters/roads_only.png"
    satellite_path = "figures/rasters/esri_satellite.png"
    osm_path = "figures/rasters/osm_map.png"
    
    try:
        # Process the raster files
        grid.load_and_process_rasters(roads_path, satellite_path, osm_path)
        
        # Create visualization
        grid.create_visualization('cellular_automaton_grid.png')
        
        # Show statistics
        stats = grid.get_grid_statistics()
        
        # Save grid data
        grid.save_grid_data()
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check the file paths:")
        print(f"Roads: {roads_path}")
        print(f"Satellite: {satellite_path}")
        print(f"OSM: {osm_path}")
    except Exception as e:
        print(f"Error: {e}")