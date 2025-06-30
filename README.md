# 🧬 Evolutionary Algorithm (EA) Project

This project explores and implements two powerful evolutionary optimization techniques — **Genetic Algorithm (GA)** and **Differential Evolution (DE)** — to solve complex function minimization problems under constraints.

It is built with educational clarity and practical insight to showcase how modern evolutionary computation techniques work under real-world optimization settings.

---

## 📘 Project Summary

In this project, we:

- Define and handle constrained optimization problems
- Implement both GA and DE from scratch
- Explore variation operators like mutation, crossover, and selection
- Manage population evolution and diversity preservation
- Tune algorithm parameters to enhance performance
- Report and analyze outcomes based on benchmark tests

> The goal is to find the **global minimum** of a benchmark function through iterative population-based refinement.

---

## ⚙️ Core Algorithms

### 🧪 Genetic Algorithm (GA)

- **Selection**: Roulette wheel selection
- **Crossover**: Single-point crossover
- **Mutation**: Gaussian mutation
- **Survivor strategy**: Elitism
- **Population size**: 50
- **Mutation rate**: 0.1
- **Crossover rate**: 0.7
- **Generations**: 100
- **Search space**: 2D solution vector [X, Y]

---

### 🔬 Differential Evolution (DE)

- **Mutation strategy**: rand/1/bin
- **Mutation factor**: 0.8
- **Crossover rate**: 0.9
- **Population size**: 50
- **Generations**: 100
- **Search space**: 2D solution vector [X, Y]

---

## 🎯 Features & Concepts

- **Constraint Handling**  
  - Penalty functions  
  - Repair methods  
  - Decoder functions  
  - Feasible space limitation

- **Parameter Tuning**
  - Manual + Optimization techniques (grid/random search)

- **Diversity Preservation**
  - Implemented **Crowding Distance** to avoid premature convergence

---

## 📈 Results

Both GA and DE were evaluated by running 100 generations with a population size of 50 over a 2D benchmark optimization space. Key metrics included:

- **Convergence speed**
- **Fitness progression**
- **Exploration vs Exploitation tradeoff**
# 🔬 Evolutionary Algorithms: Differential Evolution vs. Genetic Algorithm

This project implements and compares two powerful evolutionary optimization algorithms: **Differential Evolution (DE)** and **Genetic Algorithm (GA)**. Both algorithms are applied to a suite of benchmark functions to explore their performance in solving complex, non-linear optimization problems.

## 📌 Benchmark Functions

- 🌀 **Ackley**
- ![image](https://github.com/user-attachments/assets/b28228bc-07c1-4b8e-b81c-e5e7c3ecce60)
- 🏔️ **Bukin**
- ![image](https://github.com/user-attachments/assets/d9792ef9-1a81-4967-a636-fa2558f841f5)

- ✖️ **Cross-in-Tray**
- ![image](https://github.com/user-attachments/assets/505dd712-a106-44d1-bf51-1407e15d3ce4)

- 🌊 **Drop-Wave**
- ![image](https://github.com/user-attachments/assets/5ccb0a98-362d-498c-b9f1-c44ea6c09313)

- 🍳 **EggHolder**
- ![image](https://github.com/user-attachments/assets/c1e72a24-2b29-43c7-8a08-9543ce3a752b)


Each function is visualized in both **2D** and **3D** plots, highlighting the best solution found by each algorithm on the function’s surface.

---

## ✅ Features

- 🔁 Differential Evolution (DE) implementation
- 🧬 Genetic Algorithm (GA) with:
  - Crossover
  - Mutation
  - Roulette Wheel Selection
- 🧠 Optimization over 5 well-known benchmark functions
- 📈 2D & 3D function visualization using **Matplotlib**
- 🐍 Fully implemented in **Python** with **NumPy**

---

## 📊 Output

- Prints **best fitness per generation** for each algorithm
- Displays **2D contour plots** and **3D surface plots**
- Shows best-found solution points for DE and GA on each function
![image](https://github.com/user-attachments/assets/bb19fee0-688d-4fbe-97ef-ac53da8748ee)

---
## 📎 License

This project is open-source under the MIT License.

---

## 🔗 References

Benchmark function definitions were adapted from the collection maintained by SigOpt and Surjanovic & Bingham (Simon Fraser University):  
🔗 [https://www.sfu.ca/~ssurjano/](https://www.sfu.ca/~ssurjano/)
