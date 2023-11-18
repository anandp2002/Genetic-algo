import AG
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randint
import random

# Load Images
image = "Lenna0.05"
AG.img_noise = mpimg.imread(image + ".jpg", 1)

# Global Variables
population = AG.firstPopulation(12)
fit_first_pop = AG.computePerfPopulation(population)

# Tracking fitness for each generation
tracker = []

# Adaptive Mechanism Parameters
initial_mutation_rate = 0.1
mutation_rate_increase_factor = 1.2
mutation_rate_decrease_factor = 0.8
adaptation_interval = 5
target_fitness_improvement = 0.1

# Main Genetic Algorithm Loop
i = 0
while i <= 20:
    sorted_population = AG.computePerfPopulation(population)
    tracker.append(sorted_population[0][1])

    # Adaptive mechanism every 'adaptation_interval' generations
    if i % adaptation_interval == 0 and i > 0:
        current_fitness = sorted_population[0][1]
        previous_fitness = tracker[-adaptation_interval]

        fitness_improvement = current_fitness - previous_fitness

        # Adjust mutation rate based on fitness improvement
        if fitness_improvement > target_fitness_improvement:
            AG.mutation_rate = max(0.01, AG.mutation_rate * mutation_rate_decrease_factor)
        else:
            AG.mutation_rate = min(1.0, AG.mutation_rate * mutation_rate_increase_factor)

    parents = AG.selectFromPopulation(sorted_population, 2, 4)
    population = AG.next_generation(parents)
    AG.mutatePopulation(population, 20)
    print(i)
    i += 1


best = sorted_population[0][0]


img_denoise = cv2.fastNlMeansDenoising(AG.img_noise, None, best[0], best[1], best[2])

# Calculate PSNR between original and denoised image
psnr_original = cv2.PSNR(AG.img_noise, AG.img_noise)
psnr_denoised = cv2.PSNR(AG.img_noise, img_denoise)

print(f"PSNR Original: {psnr_original} dB")
print(f"PSNR Denoised: {psnr_denoised} dB")

cv2.imshow("denoised", img_denoise)
cv2.imshow("orginal", AG.img_noise)
cv2.imwrite(image + '_filtro.jpeg', img_denoise)

#  original
fig = plt.figure(num='Results')
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Original")
ax1.imshow(AG.img_noise)
plt.axis('off')

# denoised
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("denoised")
ax3.imshow(img_denoise)
plt.axis('off')

plt.figure(num='Fitness History')
plt.plot(tracker)
plt.ylabel('Fitness')
plt.xlabel('Number of Generations')
plt.show()

