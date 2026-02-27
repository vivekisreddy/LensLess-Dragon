# LenslessImaging

Collaborative repository for a lensless drone application. This project explores perception, sensing, and autonomy pipelines using lensless imaging and related computational methods for aerial robotics.

---

## Repository Purpose

This repository is intended to:

* Support collaborative development across multiple contributors
* Enable experimentation with perception, depth estimation, and control for lensless drone platforms
* Maintain reproducible environments across macOS and Windows

---

## Prerequisites

Before cloning and working with this repository, ensure you have the following installed:

### Required

* **Git** (latest stable version)
* **Python 3.9+** (Python 3.10 recommended)
* **pip** (bundled with Python)

### Recommended

* **Conda / Miniconda** (for environment management)
* **VS Code** or another modern IDE

---

## Cloning the Repository

```bash
git clone https://github.com/<organization-or-username>/Lensless-Drone-JP.git
cd Lensless-Drone-JP
```

---

## Environment Setup (Strongly Recommended)

To avoid dependency conflicts, **always use a virtual environment** when adding or running code with new packages.

### Option A: Conda (Recommended for Cross-Platform)

#### macOS / Windows

```bash
conda create -n lensless-drone python=3.10
conda activate lensless-drone
pip install -r requirements.txt
```

To deactivate:

```bash
conda deactivate
```

---

### Option B: Python venv (Lightweight Alternative)

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To deactivate:

```bash
deactivate
```

---

## Installing Dependencies

All shared dependencies should be listed in:

```
requirements.txt
```

When adding new packages:

1. Install them **inside your environment**
2. Update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

Do **not** install packages globally.

---

## Platform Notes

### macOS (Apple Silicon)

* Prefer Conda environments for better compatibility
* Some packages may require ARM64-specific builds

### Windows

* Use PowerShell or Anaconda Prompt
* Ensure Python is added to PATH

---

## Development Guidelines

* Create feature branches for new work:

```bash
git checkout -b feature/your-feature-name
```

* Commit frequently with clear messages
* Avoid committing:

  * Virtual environments (`venv/`, `env/`)
  * Large datasets
  * Build artifacts

---

## Repository Structure (High-Level)

```
Lensless-Drone-JP/
├── data/               # Datasets (gitignored if large)
├── src/                # Core source code
├── experiments/        # Prototypes and experiments
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore
```

---

## Reproducibility

To ensure reproducibility:

* Always specify package versions
* Document assumptions in code comments
* Log experiment configurations where applicable

---

## Contribution Expectations

All contributors are expected to:

* Use isolated environments
* Keep code platform-agnostic when possible
* Document non-obvious design decisions


