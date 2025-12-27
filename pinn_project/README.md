# Multi-Method Validation Framework for Teaching Radiation Transport

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)

**An open educational resource for teaching computational physics validation through radiation transport.**

> This repository accompanies the paper: *"Multi-Method Validation Framework for Teaching Radiation Transport: Comparing Monte Carlo, Analytical, and Physics-Informed Machine Learning Approaches"* submitted to the European Journal of Physics.

---

## 🎯 Educational Purpose

Students in computational physics courses increasingly rely on sophisticated simulation tools and machine learning models, yet often treat these as "black boxes" without developing the critical skill of **multi-method validation**. This framework teaches:

- **The validation mindset**: Why computational results must be verified against multiple independent methods
- **Method complementarity**: When Monte Carlo, analytical, and ML approaches are appropriate
- **Physics-AI integration**: How domain knowledge improves machine learning predictions
- **Critical interpretation**: Why model disagreement reveals physics, not just error

### Target Audience

- Advanced undergraduate (3rd-4th year) physics students
- Early MSc students in computational physics
- Courses: Computational physics, nuclear instrumentation, radiation physics

### Prerequisites

- Python programming (intermediate)
- Basic particle physics concepts (energy loss, stopping power)
- Familiarity with either Monte Carlo methods OR neural networks

---

## 📁 Repository Structure

```
pinn_project/
├── src/                          # Core Python modules
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── baseline_nn.py            # Standard neural network
│   ├── advanced_pinn.py          # Physics-Informed Neural Network
│   └── uncertainty_pinn.py       # PINN with uncertainty quantification
│
├── models/                       # Pre-trained models
│   ├── baseline_nn.keras         # Trained baseline model
│   └── advanced_pinn.keras       # Trained physics-informed model
│
├── notebooks/                    # Jupyter notebooks for classroom activities
│   ├── 01_monte_carlo_exploration.ipynb
│   ├── 02_nist_comparison.ipynb
│   └── 03_pinn_investigation.ipynb
│
├── results/                      # Experimental results
│   ├── ablation_study.csv        # Physics constraint ablation
│   └── hyperparameter_search.csv # Systematic constraint selection
│
├── paper/                        # Manuscript and figures
│   ├── main.tex                  # LaTeX source
│   ├── main.pdf                  # Compiled manuscript
│   └── figures/                  # Publication figures
│
├── train_baseline.py             # Train baseline neural network
├── train_advanced_pinn.py        # Train physics-informed model
├── train_ablation.py             # Run ablation experiments
├── final_comparison.py           # Compare all models
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Yash-Programmer/pinn-radiation-validation.git
cd pinn-radiation-validation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Use Pre-trained Models (Fastest)

```python
from tensorflow import keras
import numpy as np

# Load pre-trained physics-informed model
model = keras.models.load_model('models/advanced_pinn.keras')

# Predict stopping power for 200 MeV proton at 2mm depth
# Input: [Energy(MeV), Depth(mm), beta, gamma, momentum(MeV/c)]
X = np.array([[200, 2.0, 0.566, 1.213, 612.8]])
prediction = model.predict(X)
print(f"Predicted dE/dx: {prediction[0][0]:.4f} MeV/mm")
```

### 3. Train Models from Scratch

```bash
# Train baseline neural network (~5 min)
python train_baseline.py

# Train physics-informed model (~15 min)
python train_advanced_pinn.py

# Run ablation study (~30 min)
python train_ablation.py

# Compare all models
python final_comparison.py
```

---

## 📓 Classroom Activities

### Activity 1: Monte Carlo Exploration (2 hours)
**Learning objective**: Understand how Monte Carlo simulation captures particle transport physics.

Open `notebooks/01_monte_carlo_exploration.ipynb` to:
- Analyze pre-computed TOPAS simulation outputs
- Identify Bragg peak features in dose-depth curves
- Investigate how peak position correlates with incident energy
- Predict outcomes before modifying parameters

### Activity 2: Reference Data Comparison (1 hour)
**Learning objective**: Learn to validate simulations against authoritative reference data.

Open `notebooks/02_nist_comparison.ipynb` to:
- Extract mean stopping powers from simulation data
- Compare systematically against NIST PSTAR tabulated values
- Compute ratios across energies
- Develop physics-based explanations for discrepancies

### Activity 3: Physics-Informed ML Investigation (2 hours)
**Learning objective**: Demonstrate how physics constraints improve machine learning predictions.

Open `notebooks/03_pinn_investigation.ipynb` to:
- Evaluate pre-trained models on withheld energies
- Compare baseline vs. physics-constrained performance
- Conduct your own constraint ablation
- Discover which physics principles most benefit generalization

---

## 📊 Key Results

| Model | Test MSE | MAPE | Improvement |
|-------|----------|------|-------------|
| Baseline NN | 1.145×10⁻³ | 7.17% | — |
| **Physics-Informed NN** | **5.44×10⁻⁴** | **5.20%** | **+52.5%** |

### Ablation Study: Not All Physics Helps Equally

| Constraint | Effect | Teaching Point |
|------------|--------|----------------|
| Smoothness only | **+13.3%** | Strong physical justification |
| Gradient correlation only | **−92.5%** | Can conflict with simulation data |
| All constraints | +6.9% | Constraints can compete |

**Key lesson**: Physics constraints must themselves be validated—"adding physics" is not automatic.

---

## 📚 Citation

If you use this framework in your teaching or research, please cite:

```bibtex
@article{varshney2025validation,
  title={Multi-Method Validation Framework for Teaching Radiation Transport: 
         Comparing Monte Carlo, Analytical, and Physics-Informed Machine Learning Approaches},
  author={Varshney, Yash and Agarwal, Abhimanyu},
  journal={European Journal of Physics},
  year={2025},
  note={Submitted}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Ideas for Extensions

- Add new detector materials (water, tissue-equivalent)
- Extend to 2D/3D dose distributions
- Apply to other physics domains (electromagnetic showers, neutron transport)
- Develop assessment instruments for student learning outcomes

---

## 📧 Contact

- **Yash Varshney** - [yash.gurukul12@gmail.com](mailto:yash.gurukul12@gmail.com)
- **Project Link**: [https://github.com/Yash-Programmer/pinn-radiation-validation](https://github.com/Yash-Programmer/pinn-radiation-validation)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- OpenTOPAS development community for simulation support
- NIST PSTAR database for reference stopping power data
- CERN BL4S program for inspiring this educational framework
