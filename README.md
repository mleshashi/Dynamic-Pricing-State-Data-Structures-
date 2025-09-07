# Dynamic Pricing State & Data Structures — Duopoly Track

## Overview
This repository contains a simple, robust pricing bot for the Duopoly track of the Dynamic Pricing Competition (DPC). The focus is on smart data structures and online statistics, not machine learning.

- **Main algorithm:** `duopoly.py` — Implements the DPC-compliant pricing function and state management.
- **Unit tests:** `tests.py` — Validates ring buffer rollover, online statistics, edge cases, and performance.
- **Simulator:** `simulate_local.py` — Runs a toy simulation to show state evolution and pricing decisions over multiple periods.
- **Summary:** See the included PDF for a brief explanation of design choices and rationale.

## Requirements
- Python 3.9+ (compatible with DPC container)
- Libraries: numpy (other allowed libraries: pandas, xgboost, statsmodels, scikit-learn)

## Setup: Create and Activate Python Environment

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv env-dp
   ```

2. **Activate the environment:**
   ```bash
   source env-dp/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Run Unit Tests
To check correctness and constraints:
```bash
python tests.py
```

### 2. Run the Local Simulator
To see how the pricing bot behaves over several periods:
```bash
python simulate_local.py [N]
```
Where `[N]` is optional (number of periods to simulate, default is 20).

### 3. Review the Algorithm
- Read `duopoly.py` for the main pricing logic and state management.
- Comments and docstrings explain the rationale and choices.

### 4. Read the PDF Summary
- Open the PDF for a concise explanation of:
  - Data structures (what, why, size)
  - Online update equations
  - Cold-start & edge-case handling
  - Policy and trade-offs

## Deliverables
- `duopoly.py` — Pricing function and helpers
- `tests.py` — Unit tests
- `simulate_local.py` — Toy simulator (optional)
- `summary.pdf` — Design summary (≤2 pages)

## Notes
- No custom file I/O or logging is used, per competition rules.
- All state is passed via the `information_dump` object.
- The code is designed for clarity, efficiency, and compliance with DPC constraints.

---
For any questions, see comments in the code or the PDF summary.
