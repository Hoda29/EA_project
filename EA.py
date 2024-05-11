import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

output_string = ""
class Ackley:
    @staticmethod
    def evaluate(x, y):
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = 2 
        sum1 = x**2 + y**2
        sum2 = np.cos(c * x) + np.cos(c * y)
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

class Bukin:
    @staticmethod
    def evaluate(x, y):
        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

class CrossInTray:
    @staticmethod
    def evaluate(x, y):
        return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))) + 1)**0.1

class DropWave:
    @staticmethod
    def evaluate(x, y):
        return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

class EggHolder:
    @staticmethod
    def evaluate(x, y):
        return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


class DifferentialEvolution:
    def __init__(self, objective_function, num_dimensions, population_size, mutation_factor, crossover_rate, max_generations):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = np.random.uniform(low=-5, high=5, size=(population_size, num_dimensions))

    def evolve(self):
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                target_vector = self.population[i]
                random_indices = np.random.choice(range(self.population_size), size=3, replace=False)
                a, b, c = self.population[random_indices]
                mutant_vector = a + self.mutation_factor * (b - c)

                trial_vector = np.where(np.random.rand(self.num_dimensions) < self.crossover_rate, mutant_vector, target_vector)

                target_fitness = self.objective_function.evaluate(*target_vector)
                trial_fitness = self.objective_function.evaluate(*trial_vector)
                if trial_fitness < target_fitness:
                    self.population[i] = trial_vector

            best_fitness = min(self.objective_function.evaluate(*individual) for individual in self.population)
            global output_string
            output_string += f"Generation {generation}: DE Best Fitness = {best_fitness}\n"
            print(f"Generation {generation}: DE Best Fitness = {best_fitness}")

    def get_best_solution(self):
        return min(self.population, key=lambda ind: self.objective_function.evaluate(*ind))


class GeneticAlgorithm:
    def __init__(self, objective_function, num_dimensions, population_size, mutation_rate, crossover_rate, max_generations):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = np.random.uniform(low=-5, high=5, size=(population_size, num_dimensions))

    def evolve(self):
        for generation in range(self.max_generations):

            fitness = np.array([self.objective_function.evaluate(*individual) for individual in self.population])

            parents = self.selection(fitness)

            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)

            offspring_fitness = np.array([self.objective_function.evaluate(*individual) for individual in offspring])

            self.population = self.survivor_selection(self.population, fitness, offspring, offspring_fitness)

            best_fitness = np.min(fitness)
            global output_string 
            output_string += f"Generation {generation}: GA Best Fitness = {best_fitness}\n"
            print(f"Generation {generation}: GA Best Fitness = {best_fitness}")
    
    def selection(self, fitness):
       
        probabilities = abs(fitness / np.sum(fitness))
        return self.population[np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)]

    def crossover(self, parents):
      
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.num_dimensions)
                offspring[i] = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
                offspring[i+1] = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            else:
                offspring[i] = parents[i]
                offspring[i+1] = parents[i+1]
        return offspring

    def mutation(self, offspring):
    
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_index = np.random.randint(0, self.num_dimensions)
                offspring[i][mutation_index] += np.random.normal(0, 0.1)
        return offspring

    def survivor_selection(self, population, fitness, offspring, offspring_fitness):
       
        combined_population = np.concatenate((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))
        sorted_indices = np.argsort(combined_fitness)[:self.population_size]
        return combined_population[sorted_indices]

    def get_best_solution(self):
        return min(self.population, key=lambda ind: self.objective_function.evaluate(*ind))


def plot_2d_function_and_predictions(benchmark_function, de, ga, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = benchmark_function.evaluate(X, Y)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    ax.set_title(f'{benchmark_function.__name__} Function')

    best_solution_de = de.get_best_solution()
    ax.plot(best_solution_de[0], best_solution_de[1], 'ro', label='Best DE Solution')

    best_solution_ga = ga.get_best_solution()
    ax.plot(best_solution_ga[0], best_solution_ga[1], 'go', label='Best GA Solution')

    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_3d_function_and_predictions(benchmark_function, de, ga, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = benchmark_function.evaluate(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    best_solution_de = de.get_best_solution()
    ax.scatter(best_solution_de[0], best_solution_de[1], benchmark_function.evaluate(*best_solution_de), color='r', label='Best DE Solution')

    best_solution_ga = ga.get_best_solution()
    ax.scatter(best_solution_ga[0], best_solution_ga[1], benchmark_function.evaluate(*best_solution_ga), color='g', label='Best GA Solution')

    ax.set_title(f'{benchmark_function.__name__} Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.legend()
    plt.show()

def plot_optimized_results():
    global DE
    global GA
    benchmark_function = benchmark_functions[benchmark_combobox.current()]
    mutation_rate = float(mutation_rate_entry.get())
    mutation_factor = float(mutation_factor_entry.get())
    crossover_rate_de = float(crossover_rate_de_entry.get())
    crossover_rate_ga = float(crossover_rate_ga_entry.get())

    de = DifferentialEvolution(objective_function=benchmark_function, num_dimensions=2, population_size=50, mutation_factor=mutation_factor, crossover_rate=crossover_rate_de, max_generations=100)
    ga = GeneticAlgorithm(objective_function=benchmark_function, num_dimensions=2, population_size=50, mutation_rate=mutation_rate, crossover_rate=crossover_rate_ga, max_generations=100)
    de.evolve()
    ga.evolve()
    
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, output_string)
    
    if  benchmark_function == EggHolder:
        plot_2d_function_and_predictions(benchmark_function, de, ga, x_range=(-512, 512), y_range=(-512, 512))
        plot_3d_function_and_predictions(benchmark_function, de, ga, x_range=(-512, 512), y_range=(-512, 512))
    else:
        plot_2d_function_and_predictions(benchmark_function, de, ga)
        plot_3d_function_and_predictions(benchmark_function, de, ga)
root = tk.Tk()
root.title("GA/DE Algorithm")
root.geometry("860x620") 

benchmark_functions = [Ackley, Bukin, CrossInTray, DropWave, EggHolder]

main_menu_frame = tk.Frame(root, width=910, height=780, padx=20, pady=20, bg="#130a2e")
main_menu_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

mutation_factor_label = ttk.Label(main_menu_frame, text="Mutation Factor (DE):", font=("Helvetica", 12, "bold"),background = "#130a2e",foreground="white")
mutation_factor_label.grid(row=0, column=0, padx=20, pady=10, sticky="e")
mutation_factor_entry = ttk.Entry(main_menu_frame, background="#562fd0", foreground="#562fd0")
mutation_factor_entry.grid(row=0, column=1, padx=20, pady=10)

crossover_rate_de_label = ttk.Label(main_menu_frame, text="Crossover Rate (DE):", font=("Helvetica", 12, "bold"),background = "#130a2e",foreground="white")
crossover_rate_de_label.grid(row=1, column=0, padx=20, pady=10, sticky="e")
crossover_rate_de_entry = ttk.Entry(main_menu_frame, background="#562fd0", foreground="#562fd0")
crossover_rate_de_entry.grid(row=1, column=1, padx=20, pady=10)

mutation_rate_label = ttk.Label(main_menu_frame, text="Mutation Rate (GA):", font=("Helvetica", 12, "bold"),background = "#130a2e",foreground="white")
mutation_rate_label.grid(row=2, column=0, padx=20, pady=10, sticky="e")
mutation_rate_entry = ttk.Entry(main_menu_frame, background="#562fd0", foreground="#562fd0")
mutation_rate_entry.grid(row=2, column=1, padx=20, pady=10)

crossover_rate_ga_label = ttk.Label(main_menu_frame, text="Crossover Rate (GA):", font=("Helvetica", 12, "bold"),background = "#130a2e",foreground="white")
crossover_rate_ga_label.grid(row=3, column=0, padx=20, pady=10, sticky="e")
crossover_rate_ga_entry = ttk.Entry(main_menu_frame, background="#562fd0", foreground="#562fd0")
crossover_rate_ga_entry.grid(row=3, column=1, padx=20, pady=10)

benchmark_label = ttk.Label(main_menu_frame, text="Select Benchmark Function:", font=("Helvetica", 12, "bold"),background = "#130a2e",foreground="white")
benchmark_label.grid(row=4, column=0, padx=20, pady=10, sticky="e")
benchmark_combobox = ttk.Combobox(main_menu_frame, values=[func.__name__ for func in benchmark_functions])
benchmark_combobox.grid(row=4, column=1, padx=20, pady=10)
benchmark_combobox.current(0)

plot_button = tk.Button(main_menu_frame, width=20, height=2, text="Plot Optimized Results", command=plot_optimized_results,bg = "#562fd0",fg="white")
plot_button.grid(row=6, column=0, columnspan=2, padx=20, pady=10)

output_text = tk.Text(main_menu_frame, width=100, height=20,background="#1d1045" , foreground="white")
output_text.grid(row=7, column=0, columnspan=2, padx=20, pady=10)
root.mainloop()

