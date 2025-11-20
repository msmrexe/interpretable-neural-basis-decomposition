# Neural Basis Decomposition

**Visualizing the Universal Approximation Theorem through Mechanistic Interpretability**

*Developed for the M.S. Machine Learning course.*

**Neural Basis Decomposition** is a research-oriented framework designed to demystify the internal operations of Multi-Layer Perceptrons (MLPs). While neural networks are frequently treated as opaque "black boxes" that map inputs to outputs, this project employs **Mechanistic Interpretability** techniques to reverse-engineer exactly *how* a network solves function approximation tasks.

By isolating the contributions of individual hidden neurons, this repository demonstrates that a neural network is mathematically equivalent to a summation of weighted, shifted ReLU basis functions. It visualizes the **Universal Approximation Theorem** in real-time, revealing how complex non-linear topologies (such as sine waves or polynomials) are constructed through the superposition of simple piecewise linear segments. This project shifts the focus from model performance (loss convergence) to model behavior (internal representation), providing a granular view of how neurons cooperate—and interfere—to model reality.

## Features

* **Numpy Engine (From Scratch):** A modular, object-oriented implementation of backpropagation, demonstrating the calculus of gradients without autograd libraries.
* **Basis Decomposition:** A visualization suite that breaks down the final output signal into its constituent ReLU "ghost" functions.
* **Topology Analysis:** A comparative study of "Deep Folding" vs. "Polynomial Lifting" for solving non-linearly separable classification tasks (Spiral Dataset).
* **Optimizer Benchmarking:** A rigoruous comparison of First-Order (SGD) vs. Second-Order Moment (Adam) optimization dynamics.

## Core Concepts & Techniques

* **Mechanistic Interpretability:** Reverse engineering the "algorithms" learned by individual neurons.
* **Universal Approximation Theorem:** Empirical visualization of how width allows a network to approximate any continuous function.
* **Optimization Dynamics:** Implementation of Momentum and Adaptive Moment Estimation (Adam).
* **Manifold Hypothesis:** Exploring how depth untangles data manifolds in low-dimensional space.

---

## How It Works

This project bridges the **Algebra of Deep Learning** (matrices, gradients) with the **Geometry of Deep Learning** (basis functions, manifolds).

### 1. The Geometry: Basis Decomposition

At its core, a 1-hidden-layer Neural Network using ReLU activations is simply a sum of semi-infinite line segments. We can express the output $f(x)$ as:

$$f(x) = \sum_{i=1}^{N} w_{out}^{(i)} \cdot \text{ReLU}(w_{in}^{(i)} x + b^{(i)}) + b_{out}$$

Where each neuron $i$ learns three geometric properties:
1.  **The Slope ($w_{in}$):** How steep the activation is.
2.  **The Kink ($b$):** The x-coordinate where the neuron activates ($x = -b/w_{in}$).
3.  **The Importance ($w_{out}$):** The weight and direction (positive/negative) of the neuron's contribution to the final sum.

By visualizing these components individually (see `notebooks/2_Geometric_Mechanisms.ipynb`), we can observe the network "fitting" the curve piece-by-piece.

### 2. The Algebra: Optimization

The `src/numpy_engine` builds the optimization logic from first principles. We compare three update rules:

**SGD:**

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

**Momentum (Velocity Smoothing):**

$$v_{t+1} = \beta v_t + (1-\beta)\nabla L(w_t)$$
$$w_{t+1} = w_t - \eta v_{t+1}$$

**Adam (Adaptive Moments):**
Adam adapts the learning rate for each parameter individually using the first moment $m$ (mean) and second moment $v$ (uncentered variance).

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 3. The Topology: Depth vs. Width

To solve the **Spiral Classification** problem (non-linearly separable), we compare two strategies:
* **Width (Polynomial Features):** "Lifting" the 2D data into high-dimensional space ($x, y, x^2, xy, y^2, \dots$) where it becomes linearly separable.
* **Depth (MLP):** Using layers to "fold" the 2D space itself, allowing a linear cut to separate the classes without manual feature engineering.

---

## Project Structure

```
interpretable-neural-basis-decomposition/
├── .gitignore                       # Exclusions
├── LICENSE                          # MIT License
├── README.md                        # Documentation
├── requirements.txt                 # Dependencies
├── configs/
│   └── default_config.yaml          # Hyperparameters
├── notebooks/
│   ├── 1_Algebraic_Dynamics.ipynb   # From Scratch: Gradients & Optimizers
│   └── 2_Geometric_Mechanisms.ipynb # PyTorch: Basis Functions & Topology
├── scripts/
│   ├── train_optimizer_benchmark.py # CLI: Compare SGD/Momentum/Adam
│   └── visualize_basis.py           # CLI: Generate Basis Function Plot
└── src/
    ├── __init__.py
    ├── utils.py                    # Logger & Seeding
    ├── data_loader.py              # Sine, Spiral, & MNIST Generators
    ├── visualization.py            # Plotting Logic
    ├── numpy_engine/               # [FROM SCRATCH]
    │   ├── layers.py               # Linear Layer
    │   ├── activations.py          # ReLU, Sigmoid, Tanh, etc.
    │   ├── loss.py                 # CrossEntropy, MSE
    │   ├── optimizers.py           # SGD, Momentum, Adam
    │   └── network.py              # MLP Container
    └── torch_engine/               # [INTERPRETABILITY]
        ├── models.py               # ExplainableReLUNet
        └── analysis.py             # Basis extraction logic
```

## How to Use

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/msmrexe/interpretable-neural-basis-decomposition.git](https://github.com/msmrexe/interpretable-neural-basis-decomposition.git)
    cd interpretable-neural-basis-decomposition
    pip install -r requirements.txt
    ```

2.  **Run the Optimizer Benchmark (Numpy Engine):**
    Train an MLP on MNIST from scratch using different optimizers to compare convergence.

    ```bash
    python scripts/train_optimizer_benchmark.py --dataset mnist --optimizer all
    ```

3.  **Visualize Basis Decomposition (PyTorch Engine):**
    Train a network to approximate a sine wave and visualize the hidden neurons.

    ```bash
    python scripts/visualize_basis.py --hidden_dim 15 --epochs 1000
    ```

4.  **Explore the Notebooks:**
    For the deep dive into the math and step-by-step tutorials, run:

    ```bash
    jupyter notebook notebooks/
    ```

-----

## Author

Feel free to connect or reach out if you have any questions\!

  * **Maryam Rezaee**
  * **GitHub:** [@msmrexe](https://github.com/msmrexe)
  * **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

-----

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
