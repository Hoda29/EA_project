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
