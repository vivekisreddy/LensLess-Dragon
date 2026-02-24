import numpy as np
from pathlib import Path

# Configuration
input_file = Path("FisherMask_HeightMap.txt")
output_file = Path("fisher_mask_manifold.stl")
total_scale = 1000
base = 5e-4*total_scale
xy_pitch = 135e-6 * total_scale  # 135 micrometers in meters
z_scale = 1.0 * total_scale * 1000      # Scale factor for height values
base_height = 0.0  # Base height for the mesh bottom

def write_triangle(f, v1, v2, v3):
    """Write a single triangle to STL file"""
    # Calculate normal vector
    normal = np.cross(v2 - v1, v3 - v1)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 1e-10:
        normal = normal / norm_mag
    else:
        normal = np.array([0, 0, 1])  # Default normal for degenerate triangles
    
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

# Total physical dimensions
total_width = cols * xy_pitch
total_height = rows * xy_pitch
print(f"Physical dimensions: {total_width*1e3:.3f} mm x {total_height*1e3:.3f} mm")

# Create vertex grid - we need vertices at cell corners for proper connectivity
# Each cell is defined by 4 corner vertices
vertex_rows = rows + 1
vertex_cols = cols + 1

# Create vertex positions and heights
vertices = np.zeros((vertex_rows, vertex_cols, 3))

# Fill vertex positions
for i in range(vertex_rows):
    for j in range(vertex_cols):
        x = j * xy_pitch
        y = i * xy_pitch
        
        # Determine height at this vertex by averaging surrounding cells
        # Handle edge cases where vertex is at boundary
        heights = []
        for di in [-1, 0]:
            for dj in [-1, 0]:
                cell_i = i + di
                cell_j = j + dj
                if 0 <= cell_i < rows and 0 <= cell_j < cols:
                    heights.append(height_map[cell_i, cell_j])
        
        if heights:
            z = np.mean(heights)
        else:
            z = 0  # Shouldn't happen with proper bounds
            
        vertices[i, j] = [x, y, z]

# Generate STL file
print(f"Generating STL file: {output_file}")
with open(output_file, 'w') as f:
    f.write("solid fisher_mask\n")
    
    # Top surface - create triangles using shared vertices
    for i in range(rows):
        for j in range(cols):
            # Get the 4 corner vertices for this cell
            v00 = vertices[i, j]
            v01 = vertices[i, j + 1]
            v10 = vertices[i + 1, j]
            v11 = vertices[i + 1, j + 1]
            
            # Create two triangles for the top face
            write_triangle(f, v00, v01, v11)
            write_triangle(f, v00, v11, v10)
    
    # Side walls
    # Left wall (j = 0)
    for i in range(rows):
        v_top0 = vertices[i, 0]
        v_top1 = vertices[i + 1, 0]
        v_bot0 = np.array([0, i * xy_pitch, base_height])
        v_bot1 = np.array([0, (i + 1) * xy_pitch, base_height])
        
        write_triangle(f, v_top0, v_bot1, v_top1)
        write_triangle(f, v_top0, v_bot0, v_bot1)
    
    # Right wall (j = cols)
    for i in range(rows):
        v_top0 = vertices[i, cols]
        v_top1 = vertices[i + 1, cols]
        v_bot0 = np.array([total_width, i * xy_pitch, base_height])
        v_bot1 = np.array([total_width, (i + 1) * xy_pitch, base_height])
        
        write_triangle(f, v_top0, v_top1, v_bot1)
        write_triangle(f, v_top0, v_bot1, v_bot0)
    
    # Front wall (i = 0)
    for j in range(cols):
        v_top0 = vertices[0, j]
        v_top1 = vertices[0, j + 1]
        v_bot0 = np.array([j * xy_pitch, 0, base_height])
        v_bot1 = np.array([(j + 1) * xy_pitch, 0, base_height])
        
        write_triangle(f, v_top0, v_top1, v_bot1)
        write_triangle(f, v_top0, v_bot1, v_bot0)
    
    # Back wall (i = rows)
    for j in range(cols):
        v_top0 = vertices[rows, j]
        v_top1 = vertices[rows, j + 1]
        v_bot0 = np.array([j * xy_pitch, total_height, base_height])
        v_bot1 = np.array([(j + 1) * xy_pitch, total_height, base_height])
        
        write_triangle(f, v_top0, v_bot1, v_top1)
        write_triangle(f, v_top0, v_bot0, v_bot1)
    
    # Bottom surface
    for i in range(rows):
        for j in range(cols):
            x0, x1 = j * xy_pitch, (j + 1) * xy_pitch
            y0, y1 = i * xy_pitch, (i + 1) * xy_pitch
            
            v00 = np.array([x0, y0, base_height])
            v01 = np.array([x1, y0, base_height])
            v10 = np.array([x0, y1, base_height])
            v11 = np.array([x1, y1, base_height])
            
            # Bottom triangles (reversed winding for downward normal)
            write_triangle(f, v00, v11, v01)
            write_triangle(f, v00, v10, v11)
    
    f.write("endsolid fisher_mask\n")

print(f"STL file created successfully: {output_file}")
triangle_count = 2 * rows * cols + 2 * (2 * rows + 2 * cols) + 2 * rows * cols
print(f"Triangle count: {triangle_count}")