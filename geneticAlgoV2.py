import numpy as np
import cv2
import random

filter_size = 3
population_size = 100
mutation_rate = 0.2
num_generations = 500
#image to float
original_image = cv2.imread(r'D:\VSCode\Genetic-algo-main\Lenna0.05.jpg', cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.imread(r'D:\VSCode\Genetic-algo-main\Lenna_noisy.jpg', cv2.IMREAD_GRAYSCALE)

def create_random_filter(size):
    return np.array([[random.random() for _ in range(size)]for _ in range(size)])

def apply_filter(image, filter):
    return cv2.filter2D(image, -1, filter)

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def fitness(original_image, noisy_image, filter):
    filtered_image = apply_filter(noisy_image, filter)
    psnr = calculate_psnr(original_image, filtered_image)
    return psnr

def genetic_algorithm(original_image, noisy_image, filter_size, population_size, mutation_rate, num_generations):
    population = [create_random_filter(filter_size) for _ in range(population_size)]
    for generation in range(num_generations):
        fitness_scores = [fitness(original_image, noisy_image, filter) for filter in population]
        selected_indices = np.argsort(fitness_scores)[-population_size // 2:]
        parents = [population[i] for i in selected_indices]

        offspring = []
        while len(offspring) < population_size:
            p1 = random.choice(selected_indices)
            p2 = random.choice(selected_indices)
            parent1 = population[p1]
            parent2 = population[p2]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            offspring.append(child)

        population = offspring
    best_filter = population[np.argmax(fitness_scores)]
    return best_filter

def crossover(parent1, parent2):
    crossover_point = np.random.randint(0,filter_size)
    child1 = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)
    return child1

def mutate(filter, mutation_rate):
    mutated_filter = filter.copy()
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            if np.random.rand() < mutation_rate:
                mutated_filter[i, j] += np.random.random()
    return mutated_filter

best_filter = genetic_algorithm(original_image, noisy_image, filter_size, population_size, mutation_rate, num_generations)
print(best_filter)
filtered_image = apply_filter(noisy_image, best_filter)
psnr = calculate_psnr(original_image, filtered_image)
print("PSNR of filtered image:", psnr)
cv2.imshow('Original Image', original_image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
