# Parallel Multi-Coordinate Descent for Full Configuration Interaction

This repository contains the implementation of the parallel multi-coordinate descent algorithm introduced in the paper "[Parallel Multi-Coordinate Descent Methods for Full Configuration Interaction](https://arxiv.org/abs/2411.07565)". The algorithm is designed for efficient electronic structure calculations, offering a scalable approach for solving the FCI problem on shared-memory systems.

## Features
- Modified block coordinate descent with deterministic compression.
- High parallel efficiency (up to 79.3% on 128 cores).
- Supports large systems, e.g., chromium dimer (48 electrons, 42 orbitals).

## Requirements
- G++ Compiler with C++14 and OpenMP support.
- OpenBLAS & LAPACK libraries.

## Usage
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/ninotreve/mcdfci.git
   cd mcdfci
   ```
2. Ensure OpenBLAS and LAPACK are installed and set the paths:
   ```bash
   export LIBRARY_PATH=/path/to/openblas/lib:/path/to/lapack/lib:$LIBRARY_PATH
   export LD_LIBRARY_PATH=/path/to/openblas/lib:/path/to/lapack/lib:$LD_LIBRARY_PATH
   ```
3. Compile the code:
   ```bash
   make
   ```
   This will generate an executable named ```cdfci_omp```.
4. Run the program with an input file:
   ```bash
   ./cdfci_omp example_h2o_sto3g/input.json
   ```

## Input File Format
The input of the executable is a JSON file. A sample input file (`example_h2o_sto3g/input.json`) is provided and includes:

- Hamiltonian Settings: Path to the FCIDUMP file and a threshold for FCIDUMP file compression.
- Solver Settings: Parameters controlling the coordinate descent algorithm.

```json
{
    "hamiltonian": {
        "fcidump_path": "example_h2o_sto3g/FCIDUMP",
        "threshold": 0.0
    },
    "solver":{
        "cdfci": {
            "num_iterations": 20000,
            "report_interval": 1000,
            "z_threshold": 0,
            "z_threshold_search": false,
            "max_memory": 0.005,
            "coordinate_pick": "gcd_grad",
            "coordinate_update": "eig",
            "num_of_coordinate": 8,
            "estimate": 8
        }
    }
}
```

## Reproducing Experiment Results
The methods to reproduce the experiments presented in the paper are detailed in the `experiments` directory. Each subdirectory corresponds to a specific test case, including example input files, configurations, and scripts.

Feel free to open an issue for further questions or clarifications regarding the experiments.
