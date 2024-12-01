#SCA implementacija za time-series

import math
from utilities.solutionTS import Solution
from utilities.transformation import *
import random  # all random should be updated by numpy (faster calculation)
import copy
import numpy as np

class SCA:
    def __init__(self, n, function):

        self.N = n # population size
        self.function = function
        self.population = []
        self.best_solution = [None] * self.function.D #
        self.FFE = self.N
        
    
        self.dest_pos = np.zeros(self.function.D)
        self.dest_score = float("inf")



    def initial_population(self):
        for i in range(0, self.N):
            local_solution = Solution(self.function)
            self.population.append(local_solution)

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = copy.deepcopy(self.population[0].x)



    def update_position(self, t, max_iter):
        
        a = 2
        # r1 decreases linearly from a to 0
        r1 = a - t * ((a) / max_iter)
        for i in range(self.N):
             
            fitness = self.population[i].objective_function
            Xcurr = copy.deepcopy(self.population[i].x)
            
            if fitness < self.dest_score:
                self.dest_score = fitness
                self.dest_pos = Xcurr
# =============================================================================
# position update
# =============================================================================        
        for i in range(self.N):
            Xcurr = copy.deepcopy(self.population[i].x)
            Xnew = [None] * self.function.D 
            for j in range(self.function.D):
                
                # Update r2, r3, and r4 for Eq. 3.3
                r2 = (2 * np.pi) * np.random.random()
                r3 = 2 * np.random.random()
                r4 = np.random.random()
                
                # Eq. 3.3
                if r4 < (0.5):
                    # Eq. 3.1
                    Xnew[j] = Xcurr[j] + (r1 * np.sin(r2) * np.abs(r3 * self.dest_pos[j] - Xcurr[j]))
                else:
                    # Eq. 3.2
                    Xnew[j] = Xcurr[j] + (r1 * np.cos(r2) * np.abs(r3 * self.dest_pos[j] - Xcurr[j]))                

    # =============================================================================
# check boundaries
# ============================================================================= 
               #if Xnew[j] < self.function.lb[j]:
                   # Xnew[j] = self.function.lb[j]
                    
               # elif Xnew[j] > self.function.ub[j]:
                    #Xnew[j] = self.function.ub[j]

# =============================================================================
# generate new solution and compare it to the old solution
# ============================================================================= 
            # must convert to numpy array and check boundaries and apply transfer function
            Xnew = np.array(Xnew)

            # konvertujemo sve elmente koji treba da budu integer u integer
            Xnew = convertToInt(Xnew, self.function.intParams)

            #Xnew = convertToInt(Xnew, self.function.intParams)

            Xnew = self.checkBoundaries(Xnew)

            Xnew = convertToInt(Xnew, self.function.intParams)

            self.FFE = self.FFE + 1
            solution = Solution(self.function, Xnew)
         
            if solution.objective_function < self.population[i].objective_function:

                self.population[i] = solution

            
    def sort_population(self):

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return (self.population[0].objective_function,self.population[0].overallResults,self.population[0].allResults,
                self.population[0].overallResults1, self.population[0].allResults1,self.population[0].results,self.population[0].model)
    
    def get_global_worst(self):
        return (self.population[-1].objective_function,self.population[-1].overallResults,self.population[-1].allResults,
                self.population[-1].overallResults1, self.population[-1].allResults1,self.population[-1].results,self.population[-1].model)

    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'SCA'
    
    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[i].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
        
       
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[i].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):

              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
           #   print('fitness:{}'.format(self.population[i].fitness))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')

    def get_global_best_params(self):
        return self.population[0].x


    def getFFE(self):
        #metoda za prikaz broja FFE
        return self.FFE

    # funkcija koja vraca najbolji global_best_solution, ali se cuva i diversifikacija u poislednjoj populaciji
    def get_global_best_solution(self):
        # ovde pravimo liste sa objective i indicator za celu populaciji

        indicator_list = []  # ovo je indikator, sta god da je u pitanju
        objective_list = []  # ovo je objective, sta god da je u pitanju
        objective_indicator_list = []
        for i in range(len(self.population)):
            indicator_list.append(self.population[
                                      i].overallResults[0])  # ovo je za indikator, u ovom slucaju R2, ali moze da se menja
            objective_list.append(self.population[i].objective_function)  # ovo je objective, sta god da je
        objective_indicator_list.append(objective_list)
        objective_indicator_list.append(indicator_list)
        self.population[0].diversity = objective_indicator_list

        return self.population[0]

    def checkBoundaries(self, Xnew):
        for j in range(self.function.D):
            if Xnew[j] < self.function.lb[j]:
                Xnew[j] = self.function.lb[j]

            elif Xnew[j] > self.function.ub[j]:
                Xnew[j] = self.function.ub[j]
        return Xnew