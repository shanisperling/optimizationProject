from __future__ import division
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
from time import time
from memory_profiler import memory_usage


##### 3 datasets #####
Oliver30  = [(54, 67),(54, 62),(37, 84),(41, 94),(2, 99),(7, 64),(25, 62),(22, 60),(18, 54),(4, 50),(13, 40),(18, 40),(24, 42),(25, 38),(44, 35),(41, 26),(45, 21),(58, 35),(62, 32),(82,  7),(91, 38),(83, 46),(71, 44),(64, 60),(68, 58),(83, 69),(87, 76),(74, 78),(71, 71),(58, 69)]
Eil51 = [(37, 52), (49, 49), (52 ,64), (20, 26), (40, 30), (21, 47), (17, 63), (31, 62), (52, 33), (51, 21), (42, 41), (31, 32),(5, 25), (12, 42), (36, 16), (52, 41), (27, 23), (17, 33), (13, 13), (57, 58), (62, 42), (42, 57), (16, 57), (8, 52), (7, 38),(27, 68), (30, 48), (43, 67), (58, 48), (58, 27), (37, 69), (38, 46), (46, 10), (61, 33), (62, 63), (63, 69), (32, 22), (45,35),(59, 15), (5, 6), (10, 17), (21, 10), (5, 64), (30, 15), (39, 10), (32, 39), (25, 32), (25, 55), (48, 28), (56, 37), (30, 40)]
Berlin52 = [(565,575),(25,185),(345,750),(945,685),(845,655),(880,660),(25,230),(525,1000),(580,1175),(650,1130),(1605,620), 
(1220,580),(1465,200),(1530,5),(845,680),(725,370),(145,665),(415,635),(510,875), (560,365),(300,465),(520,585),
(480,415),(835,625),(975,580),(1215,245),(1320,315),(1250,400),(660,180),(410,250),(420,555),(575,665),(1150,1160),
(700,580),(685,595),(685,610),(770,610),(795,645),(720,635),(760,650),(475,960),(95,260),(875,920),(700,500),
(555,815),(830,485),(1170,65),(830,610),(605,625),(595,360),(1340,725),(1740,245)]

# choose any dataset.
dataset = Oliver30
array = np.zeros((len(dataset), 2))
for i in range(len(dataset)):
    array[i][0] = dataset[i][0]
    array[i][1] = dataset[i][1]
num_points = len(dataset)
distance_matrix = spatial.distance.cdist(array, array, metric='euclidean')

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

###### shortest path #######
def checkShortestPath():
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=50, max_iter=70,
                  distance_matrix=distance_matrix,
                   alpha=1, beta=3, rho=0.3)
    best_x, best_y = aca.run()
    print("best_x",best_x )
    print("best_y",best_y )

    # Plot the result
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = array[best_points_, :]
    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.savefig("pic_ACA.png")

###### run time #######
def checkRunTime():
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=50, max_iter=70,
                  distance_matrix=distance_matrix,
                   alpha=1, beta=3, rho=0.3)
    t0 = time()
    best_x, best_y = aca.run()
    t1 = time()
    print("time =" , t1-t0)

###### memory usage #######
def checkMemory():
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=50, max_iter=70,
                  distance_matrix=distance_matrix,
                   alpha=1, beta=3, rho=0.3)
    mem_usage = memory_usage(aca.run)
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))

def main():
    # prints the shortest path and the path with ant colony algorithm 
    checkShortestPath()
    # prints the run time of the ant colony algorithm
    checkRunTime()
    # prints the max memory usage
    checkMemory()

if __name__ == "__main__":
    main()