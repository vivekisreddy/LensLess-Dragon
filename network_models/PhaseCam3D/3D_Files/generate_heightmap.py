import numpy as np
from pathlib import Path

# --- Config ---
in_path = Path("FisherMask_HeightMap.txt")
cell_pitch = 135e-6  # meters per grid cell in X and Y
z_scale = 1.0        # multiply heights if you want to exaggerate Z
base_z = 0.0         # where to close the mesh (meters)
out_path = Path("heightmap_mesh.stl")

# --- Load height map ---
H = np.loadtxt(in_path).astype(np.float64) * z_scale
ny, nx = H.shape
xs = np.arange(nx, dtype=np.float64) * cell_pitch
ys = np.arange(ny, dtype=np.float64) * cell_pitch

def tri_str(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    n = n / (np.linalg.norm(n) or 1.0)
    return (
        f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}\n"
        "    outer loop\n"
        f"      vertex {v0[0]:.8e} {v0[1]:.8e} {v0[2]:.8e}\n"
        f"      vertex {v1[0]:.8e} {v1[1]:.8e} {v1[2]:.8e}\n"
        f"      vertex {v2[0]:.8e} {v2[1]:.8e} {v2[2]:.8e}\n"
        "    endloop\n"
        "  endfacet\n"
    )

facets = []

# Top surface
for j in range(ny - 1):
    y0, y1 = ys[j], ys[j + 1]
    for i in range(nx - 1):
        x0, x1 = xs[i], xs[i + 1]
        z00, z10 = H[j, i], H[j, i + 1]
        z01, z11 = H[j + 1, i], H[j + 1, i + 1]
        v00 = np.array([x0, y0, z00])
        v10 = np.array([x1, y0, z10])
        v01 = np.array([x0, y1, z01])
        v11 = np.array([x1, y1, z11])
        facets.append(tri_str(v00, v10, v11))
        facets.append(tri_str(v00, v11, v01))

# West wall
i = 0; xw = xs[i]
for j in range(ny - 1):
    y0, y1 = ys[j], ys[j + 1]
    v_top0 = np.array([xw, y0, H[j, i]])
    v_top1 = np.array([xw, y1, H[j + 1, i]])
    v_bot0 = np.array([xw, y0, base_z])
    v_bot1 = np.array([xw, y1, base_z])
    facets.append(tri_str(v_top1, v_top0, v_bot0))
    facets.append(tri_str(v_top1, v_bot0, v_bot1))

# East wall
i = nx - 1; xe = xs[i]
for j in range(ny - 1):
    y0, y1 = ys[j], ys[j + 1]
    v_top0 = np.array([xe, y0, H[j, i]])
    v_top1 = np.array([xe, y1, H[j + 1, i]])
    v_bot0 = np.array([xe, y0, base_z])
    v_bot1 = np.array([xe, y1, base_z])
    facets.append(tri_str(v_top0, v_top1, v_bot0))
    facets.append(tri_str(v_top1, v_bot1, v_bot0))

# South wall
j = 0; ys0 = ys[j]
for i in range(nx - 1):
    x0, x1 = xs[i], xs[i + 1]
    v_top0 = np.array([x0, ys0, H[j, i]])
    v_top1 = np.array([x1, ys0, H[j, i + 1]])
    v_bot0 = np.array([x0, ys0, base_z])
    v_bot1 = np.array([x1, ys0, base_z])
    facets.append(tri_str(v_top1, v_top0, v_bot0))
    facets.append(tri_str(v_top1, v_bot0, v_bot1))

# North wall
j = ny - 1; yn = ys[j]
for i in range(nx - 1):
    x0, x1 = xs[i], xs[i + 1]
    v_top0 = np.array([x0, yn, H[j, i]])
    v_top1 = np.array([x1, yn, H[j, i + 1]])
    v_bot0 = np.array([x0, yn, base_z])
    v_bot1 = np.array([x1, yn, base_z])
    facets.append(tri_str(v_top0, v_top1, v_bot0))
    facets.append(tri_str(v_top1, v_bot1, v_bot0))

# Bottom plate at base_z
for j in range(ny - 1):
    y0, y1 = ys[j], ys[j + 1]
    for i in range(nx - 1):
        x0, x1 = xs[i], xs[i + 1]
        v00 = np.array([x0, y0, base_z])
        v10 = np.array([x1, y0, base_z])
        v01 = np.array([x0, y1, base_z])
        v11 = np.array([x1, y1, base_z])
        facets.append(tri_str(v11, v10, v00))  # normal downward
        facets.append(tri_str(v01, v11, v00))

with open(out_path, "w") as f:
    f.write("solid heightmap\n")
    f.writelines(facets)
    f.write("endsolid heightmap\n")

print("Wrote", out_path)
