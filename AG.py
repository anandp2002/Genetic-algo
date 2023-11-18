import cv2
#from skimage.measure import compare_ssim as ssim
from astropy.stats import median_absolute_deviation as mad
from random import randint
import operator
import random

mutation_rate = 0.1  # Set an initial mutation rate

# Declaring images as global variables
# noise = noise
img_noise = None

# Creating the first population; size will be the number of individuals in the population
def firstPopulation(size):
    population = []
    for i in range(size):
        # Inserting a vector of 3 positions with random values into our population
        # They will be our parameters
        population.append((randint(1, 40), randint(1, 40), randint(1, 6)))
    return population

# Function that calculates the fitness of ONE individual; the values passed are the parameters to calculate
def fitness(h, twindows, swindows):
    # img_denoise will be our image with the filter applied
    img_denoise = cv2.fastNlMeansDenoising(img_noise, None, h, twindows, swindows)
    # Then we take this image with the filter applied and measure its noise level
    fit = 1 / mad(img_denoise, axis=None)
    # fit will be the fitness value of each individual
    return fit

# Function that calculates the fitness of the ENTIRE population
def computePerfPopulation(population):
    populationPerf = {}
    for i in population:
        # 'i' will receive each individual from our population
        # We take the values of the 3 parameters of 'i' and pass them to our fitness function
        populationPerf[i] = fitness(i[0], i[1], i[2])
    # A vector organized from highest to lowest fitness will be returned
    return sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=True)

# Function to select individuals from the population
# best_sample = number of good individuals to select
# lucky_few = number of random individuals to pick
def selectFromPopulation(populationSorted, best_sample, lucky_few):
    nextGeneration = []
    # Inserting the best individuals into the parents' population
    for i in range(best_sample):
        nextGeneration.append(populationSorted[i][0])
    # Inserting random individuals into the parents' population
    for i in range(lucky_few):
        nextGeneration.append(random.choice(populationSorted)[0])
    random.shuffle(nextGeneration)
    return nextGeneration

# Function to create a child by combining chromosomes of two individuals
def createChild(parent1, parent2):
    child = ((parent1[0], parent1[1], parent2[2]))
    # Returns a child
    return child

# Function to get the entire population of parents and generate children
def next_generation(population):
    ng = population
    # Iterates through our vector of parents and performs crossover
    for i in range(0, (len(population) - 1), 2):
        ng.append(createChild(population[i], population[i + 2]))
        ng.append(createChild(population[i + 2], population[i]))
    # Returns the next generation
    return ng

# Function to mutate a gene
# Receives an individual as a parameter
def mutateGene(individual):
    aux = list(individual)
    # Randomly chooses which gene will be mutated
    index_modification = int(randint(0, len(individual) - 2))
    # 50% chance of the gene value being added or subtracted
    if random.random() * 100 < 50:
        # Adding a random value to the gene
        aux[index_modification] += randint(0, 5)
    else:
        # Subtracting a random value from the gene
        aux[index_modification] -= randint(0, 5)
    return tuple(aux)

# Function to mutate the population
# Receives the mutation chance as a parameter
def mutatePopulation(population, mutation_chance):
    for i in range(len(population)):
        if random.random() * 100 < mutation_chance:
            # If it enters the 'if' statement, an individual from our population will be mutated
            population[i] = mutateGene(population[i])

