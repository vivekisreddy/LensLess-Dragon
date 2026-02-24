import numpy as np
from pathlib import Path

# Configuration
input_file = Path("FisherMask_HeightMap.txt")
output_file = Path("fisher_mask_mfg.stl")
xy_pitch = .1  # 135 micrometers in meters
z_scale = 1.0      # Scale factor for height values
base_height = 0.0  # Base height for the mesh bottom

def write_triangle(f, v1, v2, v3):
    """Write a single triangle to STL file"""
    # Calculate normal vector
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / (np.linalg.norm(normal) + 1e-10)  # Avoid division by zero
    
    f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
    f.write("    outer loop\n")
    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
    f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")

# Load height map data
print(f"Loading height map from {input_file}...")
height_map = np.loadtxt(input_file) * z_scale
rows, cols = height_map.shape
print(f"Height map dimensions: {rows} x {cols}")

# Create coordinate arrays - each matrix element represents center of a square
# Grid goes from -pitch/2 to (cols-0.5)*pitch and -pitch/2 to (rows-0.5)*pitch
x_coords = (np.arange(cols) + 0.5) * xy_pitch
y_coords = (np.arange(rows) + 0.5) * xy_pitch

# Total physical dimensions
total_width = cols * xy_pitch
total_height = rows * xy_pitch
print(f"Physical dimensions: {total_width*1e3:.3f} mm x {total_height*1e3:.3f} mm")

# Generate STL file
print(f"Generating STL file: {output_file}")
with open(output_file, 'w') as f:
    f.write("solid fisher_mask\n")
    
    # Generate top surface triangles
    for i in range(rows):
        for j in range(cols):
            # Get center coordinates and height for this cell
            x_center = x_coords[j]
            y_center = y_coords[i]
            z_center = height_map[i, j]
            
            # Define the four corners of the current grid cell
            x0 = x_center - xy_pitch/2
            x1 = x_center + xy_pitch/2
            y0 = y_center - xy_pitch/2
            y1 = y_center + xy_pitch/2
            
            # All four corners have the same height (flat square)
            v00 = np.array([x0, y0, z_center])
            v01 = np.array([x1, y0, z_center])
            v10 = np.array([x0, y1, z_center])
            v11 = np.array([x1, y1, z_center])
            
            # Split square into two triangles
            write_triangle(f, v00, v01, v11)
            write_triangle(f, v00, v11, v10)
    
    # Generate connecting walls between adjacent cells
    # Vertical walls between horizontally adjacent cells
    for i in range(rows):
        for j in range(cols - 1):
            # Current cell and next cell to the right
            x_curr = x_coords[j]
            x_next = x_coords[j + 1]
            y_center = y_coords[i]
            z_curr = height_map[i, j]
            z_next = height_map[i, j + 1]
            
            # Shared edge between cells
            x_edge = (x_curr + x_next) / 2
            y0 = y_center - xy_pitch/2
            y1 = y_center + xy_pitch/2
            
            # Create vertical wall
            v_curr0 = np.array([x_edge, y0, z_curr])
            v_curr1 = np.array([x_edge, y1, z_curr])
            v_next0 = np.array([x_edge, y0, z_next])
            v_next1 = np.array([x_edge, y1, z_next])
            
            write_triangle(f, v_curr0, v_next1, v_curr1)
            write_triangle(f, v_curr0, v_next0, v_next1)
    
    # Horizontal walls between vertically adjacent cells
    for i in range(rows - 1):
        for j in range(cols):
            # Current cell and next cell below
            x_center = x_coords[j]
            y_curr = y_coords[i]
            y_next = y_coords[i + 1]
            z_curr = height_map[i, j]
            z_next = height_map[i + 1, j]
            
            # Shared edge between cells
            y_edge = (y_curr + y_next) / 2
            x0 = x_center - xy_pitch/2
            x1 = x_center + xy_pitch/2
            
            # Create vertical wall
            v_curr0 = np.array([x0, y_edge, z_curr])
            v_curr1 = np.array([x1, y_edge, z_curr])
            v_next0 = np.array([x0, y_edge, z_next])
            v_next1 = np.array([x1, y_edge, z_next])
            
            write_triangle(f, v_curr0, v_curr1, v_next1)
            write_triangle(f, v_curr0, v_next1, v_next0)
    
    # Generate side walls and bottom
    # Left wall (x = 0)
    for i in range(rows):
        y_center = y_coords[i]
        z_height = height_map[i, 0]
        
        y0 = y_center - xy_pitch/2
        y1 = y_center + xy_pitch/2
        
        v_top0 = np.array([0, y0, z_height])
        v_top1 = np.array([0, y1, z_height])
        v_bot0 = np.array([0, y0, base_height])
        v_bot1 = np.array([0, y1, base_height])
        
        write_triangle(f, v_top0, v_bot0, v_top1)
        write_triangle(f, v_top1, v_bot0, v_bot1)
    
    # Right wall (x = max)
    x_max = total_width
    for i in range(rows):
        y_center = y_coords[i]
        z_height = height_map[i, -1]
        
        y0 = y_center - xy_pitch/2
        y1 = y_center + xy_pitch/2
        
        v_top0 = np.array([x_max, y0, z_height])
        v_top1 = np.array([x_max, y1, z_height])
        v_bot0 = np.array([x_max, y0, base_height])
        v_bot1 = np.array([x_max, y1, base_height])
        
        write_triangle(f, v_top1, v_bot0, v_top0)
        write_triangle(f, v_bot1, v_bot0, v_top1)
    
    # Front wall (y = 0)
    for j in range(cols):
        x_center = x_coords[j]
        z_height = height_map[0, j]
        
        x0 = x_center - xy_pitch/2
        x1 = x_center + xy_pitch/2
        
        v_top0 = np.array([x0, 0, z_height])
        v_top1 = np.array([x1, 0, z_height])
        v_bot0 = np.array([x0, 0, base_height])
        v_bot1 = np.array([x1, 0, base_height])
        
        write_triangle(f, v_top1, v_bot0, v_top0)
        write_triangle(f, v_bot1, v_bot0, v_top1)
    
    # Back wall (y = max)
    y_max = total_height
    for j in range(cols):
        x_center = x_coords[j]
        z_height = height_map[-1, j]
        
        x0 = x_center - xy_pitch/2
        x1 = x_center + xy_pitch/2
        
        v_top0 = np.array([x0, y_max, z_height])
        v_top1 = np.array([x1, y_max, z_height])
        v_bot0 = np.array([x0, y_max, base_height])
        v_bot1 = np.array([x1, y_max, base_height])
        
        write_triangle(f, v_top0, v_bot0, v_top1)
        write_triangle(f, v_top1, v_bot0, v_bot1)
    
    # Bottom surface
    for i in range(rows):
        for j in range(cols):
            x_center = x_coords[j]
            y_center = y_coords[i]
            
            x0 = x_center - xy_pitch/2
            x1 = x_center + xy_pitch/2
            y0 = y_center - xy_pitch/2
            y1 = y_center + xy_pitch/2
            
            v00 = np.array([x0, y0, base_height])
            v01 = np.array([x1, y0, base_height])
            v10 = np.array([x0, y1, base_height])
            v11 = np.array([x1, y1, base_height])
            
            # Bottom triangles (reversed winding for downward normal)
            write_triangle(f, v00, v11, v01)
            write_triangle(f, v00, v10, v11)
    
    f.write("endsolid fisher_mask\n")

print(f"STL file created successfully: {output_file}")
# Updated triangle count: top surface + connecting walls + side walls + bottom
vertical_walls = 2 * rows * (cols - 1)  # walls between horizontal neighbors
horizontal_walls = 2 * (rows - 1) * cols  # walls between vertical neighbors
print(f"Triangle count: {2 * rows * cols + vertical_walls + horizontal_walls + 4 * rows + 4 * cols + 2 * rows * cols}")