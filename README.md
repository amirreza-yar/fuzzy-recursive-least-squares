# Fuzzy Recursive Least Squares (FRLS) Modeling

## Overview

This project implements a Fuzzy Recursive Least Squares (FRLS) algorithm to model systems with four inputs and one output. The FRLS algorithm integrates fuzzy logic with the Recursive Least Squares (RLS) method to handle nonlinearity and uncertainty in the data.

## Theory

### Recursive Least Squares (RLS)

RLS is an adaptive filter algorithm that recursively finds the coefficients minimizing a weighted linear least squares cost function. The RLS algorithm updates parameter estimates with each new data point, making it efficient for real-time applications. The RLS update equations ensure that the parameter vector and the covariance matrix are adjusted at each step to minimize the prediction error.

### Fuzzy Logic

Fuzzy logic deals with reasoning that is approximate rather than fixed and exact. In fuzzy systems, input variables are mapped to output variables using fuzzy sets, membership functions, and a set of rules. Fuzzy logic provides a way to handle uncertainty and imprecision in modeling complex systems.

### Fuzzy Recursive Least Squares (FRLS)

FRLS combines the adaptive nature of RLS with the flexibility of fuzzy logic. In FRLS:
- The input data and error terms are fuzzified using membership functions.
- Fuzzy rules are applied to relate the fuzzified inputs to the adjustments of the RLS parameters.
- The fuzzy inference system determines how the RLS parameters should be adjusted based on the current state of the system.
- The defuzzified adjustment values are then used to update the parameter vector, enhancing the model's ability to handle nonlinearities and uncertainties.

## Applications

FRLS can be applied in various fields such as:
- **Control Systems**: For adaptive control in nonlinear and uncertain environments.
- **Financial Modeling**: For predicting stock prices and managing investment portfolios.
- **Signal Processing**: For filtering and noise reduction in signals with non-stationary characteristics.
- **System Identification**: For modeling and predicting the behavior of complex systems in engineering and science.

## Data

The dataset contains four input variables (`x1`, `x2`, `x3`, `x4`) and one output variable (`y`). An example of the dataset is shown below:


|  x1   | x2 |  x3   |  x4  |   y   |
|-------|----|-------|-------|-------|
| 14.57 |  0 | 14.51 | 0.06  | 15.31 |
| 15.31 |  0 | 15.29 | 0.02  | 14.76 |
| 14.76 |  0 | 14.72 | 0.04  | 18.60 |


## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip3 (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/amirreza-yar/fuzzy-recursive-least-squares/
    cd fuzzy-recursive-least-squares
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    virtualenv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip3 install -r requirements.txt
    ```

### Running the Project

After installing the necessary packages, you can run the main script to see the FRLS algorithm in action. The script reads the data, applies the FRLS algorithm, and outputs the parameter estimates.

```bash
python main.py
```
Also you can see the [detailed document](notebook/rls_doc.md)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the need to model complex systems with uncertainty and nonlinearity.
- Thanks to the contributors and the open-source community for their valuable resources and support.
