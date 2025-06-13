# Perceptron Implementation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-green)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

## Overview

This project provides a comprehensive implementation of the **Perceptron algorithm**, one of the fundamental building blocks of neural networks and machine learning. The Perceptron, introduced by Frank Rosenblatt in 1957, is a binary linear classifier that serves as the foundation for understanding more complex neural network architectures.

## What is a Perceptron?

The Perceptron is a simple supervised learning algorithm that:
- Acts as a **binary linear classifier**
- Maps input vectors to binary output values using a hyperplane
- Uses a **learning algorithm** to automatically adjust weights
- Represents the simplest form of an artificial neural network

### Mathematical Foundation

The Perceptron computes:
```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
```

Where:
- `w` = weight vector
- `x` = input features
- `bias` = threshold adjustment
- `activation` = step function (0 or 1)

## Features

- **From-Scratch Implementation**: Pure Python implementation without ML libraries
- **Mathematical Visualization**: Clear plots showing decision boundaries
- **Logic Gate Learning**: Implementation on AND, OR, and XOR problems
- **Training Visualization**: Real-time plotting of learning progress
- **Binary Classification**: General binary classification capabilities
- **Educational Focus**: Well-commented code for learning purposes

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Examples](#examples)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [Resources](#resources)

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Required Libraries

```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install jupyter
pip install seaborn
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/Arshnoor-Singh-Sohi/Perceptron.git
cd Perceptron
```

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   Navigate to `perceptron.ipynb`

3. **Run all cells**:
   Execute cells sequentially to see the implementation in action

### Quick Start Example

```python
# Create a simple perceptron
perceptron = Perceptron(learning_rate=0.1, max_iterations=100)

# Train on AND gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Fit the model
perceptron.fit(X, y)

# Make predictions
predictions = perceptron.predict(X)
print(predictions)  # [0, 0, 0, 1]
```

## Project Structure

```
Perceptron/
│
├── perceptron.ipynb                # Main implementation notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── data/                          # Sample datasets (if applicable)
│   ├── logic_gates.csv
│   └── binary_classification.csv
├── images/                        # Generated plots and visualizations
│   ├── decision_boundary.png
│   ├── training_progress.png
│   └── logic_gates_results.png
└── utils/                         # Helper functions (if applicable)
    └── visualization.py
```

## Algorithm Details

### The Perceptron Learning Algorithm

1. **Initialize** weights and bias to small random values
2. **For each training example**:
   - Calculate the weighted sum: `net_input = w·x + bias`
   - Apply activation function: `output = 1 if net_input ≥ 0 else 0`
   - Update weights: `w = w + learning_rate × (target - output) × input`
   - Update bias: `bias = bias + learning_rate × (target - output)`
3. **Repeat** until convergence or maximum iterations reached

### Key Components

- **Activation Function**: Step function (Heaviside)
- **Learning Rule**: Perceptron learning rule (delta rule)
- **Weight Update**: Based on prediction errors
- **Convergence**: Guaranteed for linearly separable data

## Examples

### 1. Logic Gates Implementation

The notebook demonstrates learning of basic logic operations:

#### AND Gate
```
Input: [0,0] → Output: 0
Input: [0,1] → Output: 0  
Input: [1,0] → Output: 0
Input: [1,1] → Output: 1
```

#### OR Gate
```
Input: [0,0] → Output: 0
Input: [0,1] → Output: 1
Input: [1,0] → Output: 1
Input: [1,1] → Output: 1
```

#### XOR Problem
Demonstrates the **limitation** of single-layer perceptrons:
```
Input: [0,0] → Output: 0
Input: [0,1] → Output: 1
Input: [1,0] → Output: 1
Input: [1,1] → Output: 0
```
*Note: XOR is not linearly separable and cannot be learned by a single perceptron*

### 2. Binary Classification

Application to real-world binary classification problems with:
- Feature visualization
- Decision boundary plotting
- Performance metrics (accuracy, precision, recall)

## Limitations

The Perceptron has several important limitations:

1. **Linear Separability**: Can only learn linearly separable patterns
2. **XOR Problem**: Cannot solve the XOR problem (historically significant limitation)
3. **Binary Classification**: Limited to two-class problems
4. **No Probabilistic Output**: Outputs hard classifications, not probabilities
5. **Sensitive to Outliers**: Can be affected by noisy data

## Visualization Features

The implementation includes:

- **Decision Boundary Plots**: Visual representation of learned boundaries
- **Training Progress**: Error reduction over iterations
- **Weight Evolution**: How weights change during training
- **Data Point Classification**: Color-coded predictions

## Educational Value

This implementation serves as an excellent introduction to:

- **Machine Learning Fundamentals**
- **Neural Network Concepts**
- **Gradient Descent Optimization**
- **Binary Classification**
- **Feature Space Visualization**
- **Algorithm Limitations**

## Key Learning Outcomes

After exploring this project, you will understand:

- How neural networks learn from data
- The importance of linear separability
- Basic optimization in machine learning
- Historical context of AI development
- Foundation for multi-layer perceptrons

## Technologies Used

- **Python**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation (if used)
- **Jupyter Notebook**: Interactive development environment

## Performance Metrics

The implementation tracks:

- **Accuracy**: Overall correctness of predictions
- **Training Error**: Number of misclassifications per epoch
- **Convergence Time**: Iterations needed to reach solution
- **Weight Stability**: Final weight values

## Contributing

Contributions are welcome! Here are ways you can contribute:

### Ideas for Enhancement

- [ ] Multi-class perceptron implementation
- [ ] Pocket algorithm for non-separable data
- [ ] Comparison with other linear classifiers
- [ ] More complex datasets
- [ ] Interactive visualizations
- [ ] Performance benchmarking

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## Resources

### Further Reading

- [Original Perceptron Paper](https://psycnet.apa.org/record/1959-09865-001) - Rosenblatt, 1958
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) - Christopher Bishop

### Related Concepts

- **Multi-Layer Perceptron (MLP)**
- **Support Vector Machines (SVM)**
- **Logistic Regression**
- **Neural Network Fundamentals**
- **Gradient Descent Optimization**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Arshnoor Singh Sohi**

- GitHub: [@Arshnoor-Singh-Sohi](https://github.com/Arshnoor-Singh-Sohi)

## Acknowledgments

- Frank Rosenblatt for the original Perceptron algorithm
- Machine learning community for educational resources
- Open source contributors to NumPy and Matplotlib

---

**Historical Note**: The Perceptron represents a crucial milestone in artificial intelligence history. Despite its limitations (notably highlighted in Minsky and Papert's 1969 book "Perceptrons"), it laid the groundwork for modern neural networks and deep learning.
