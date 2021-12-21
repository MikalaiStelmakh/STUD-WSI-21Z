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
            previous_point = np.array(point)
            point -= step*gradient
            if np.array_equal(previous_point, point):
                return None
        else:
            break
        if function(point) == np.Inf:
            return None

    return iterations


def newton_method(function, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y,
                  starting_point, step, epsilon):
    iterations = 0
    point = starting_point
    previous_points = []
    while True:
        try:
            if function(point) > epsilon:
                iterations += 1
                hessian = np.array([[df_x_df_x(point), df_x_df_y(point)],
                                    [df_x_df_y(point), df_y_df_y(point)]])
                gradient = np.array([df_x(point), df_y(point)])
                hessian_inv = np.linalg.inv(hessian)
                d = hessian_inv.dot(gradient)
                if len(previous_points) == 3:
                    previous_points.pop(0)
                    previous_points.append(function(point))
                else:
                    previous_points.append(function(point))
                point -= step*d
                if function(point) in previous_points:
                    return None
            else:
                break
            if function(point) == np.Inf:
                return None
        except np.linalg.LinAlgError:
            return None

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

    Xs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    Ys = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    POINTS = list(product(Xs, Ys))
    # POINTS = [(-1, -1), (2, 2)]
    epsilon = 1e-12
    steepest_descent_step = 0.00001
    newton_step = 1.5

    result = {
        "steepest_descent_method": {"benchmarks": [{} for _ in POINTS]},
        "newton_method": {"benchmarks": [{} for _ in POINTS]}
        }
    for index, current_point in enumerate(POINTS):
        iterations = steepest_descent(f, df_x, df_y, current_point, steepest_descent_step, epsilon)
        result["steepest_descent_method"]["benchmarks"][index]["point"] = current_point
        result["steepest_descent_method"]["benchmarks"][index]["iterations"] = iterations
        # iterations = newton_method(f, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y, current_point, newton_step, epsilon)
        # result["newton_method"]["benchmarks"][index]["point"] = current_point
        # result["newton_method"]["benchmarks"][index]["iterations"] = iterations
    with open('.benchmarks/data5.json', 'w') as fp:
        json.dump(result, fp, indent=2)



