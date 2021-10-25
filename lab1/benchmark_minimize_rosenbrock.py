import numpy as np
from itertools import product
import json


def steepest_descent(function, df_x, df_y, starting_point, step, epsilon):
    iterations = 0
    point = starting_point
    while True:
        if function(point) > epsilon:
            iterations += 1
            gradient = np.array([df_x(point), df_y(point)], dtype='double')
            point -= step*gradient
        else:
            iterations -= 1
            break
    return iterations


def newton_method(function, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y,
                  starting_point, step, epsilon):
    iterations = 0
    point = starting_point
    while True:
        if function(point) > epsilon:
            iterations += 1
            hessian = np.array([[df_x_df_x(point), df_x_df_y(point)],
                                [df_x_df_y(point), df_y_df_y(point)]])
            gradient = np.array([df_x(point), df_y(point)])
            hessian_inv = np.linalg.inv(hessian)
            d = hessian_inv.dot(gradient)
            point -= step*d
        else:
            iterations -= 1
            break
    return iterations


if __name__ == "__main__":

    def f(point):
        return ((1-point[0])**2+100*(point[1]-point[0]**2)**2)

    def df_x(point):
        return -400*point[0]*(-point[0]**2+point[1])+2*point[0]-2

    def df_y(point):
        return -200*point[0]**2 + 200*point[1]

    def df_x_df_x(point):
        return 1200*point[0]**2 - 400*point[1] + 2

    def df_y_df_y(point):
        return 200

    def df_x_df_y(point):
        return -400*point[0]

    Xs = [-4, -2, 0, 2, 4]
    Ys = [-4, -2, 0, 2, 4]
    POINTS = list(product(Xs, Ys))
    # POINTS = [(-1, -1), (2, 2)]
    epsilon = 1e-12
    steepest_descent_step = 0.0001
    newton_step = 1

    result = {
        "steepest_descent_method": {"benchmarks": [{} for _ in POINTS]},
        "newton_method": {"benchmarks": [{} for _ in POINTS]}
        }
    for index, current_point in enumerate(POINTS):
        iterations = steepest_descent(f, df_x, df_y, current_point, steepest_descent_step, epsilon)
        result["steepest_descent_method"]["benchmarks"][index]["point"] = current_point
        result["steepest_descent_method"]["benchmarks"][index]["iterations"] = iterations
        iterations = newton_method(f, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y, current_point, newton_step, epsilon)
        result["newton_method"]["benchmarks"][index]["point"] = current_point
        result["newton_method"]["benchmarks"][index]["iterations"] = iterations
    with open('.benchmarks/data.json', 'w') as fp:
        json.dump(result, fp, indent=2)



