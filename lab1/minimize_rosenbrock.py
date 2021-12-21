import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d
import argparse


def steepest_descent(function, df_x, df_y, starting_point, step, iterations, epsilon,
                     ax=None, point_radius=0.1, last_point_radius=0.5,
                     point_color='red', last_point_color='blue'):
    '''Approximate solution of f(x, y)=0 by Steepest Gradient Descent method.

    Parameters
    ----------
    function : function
        Function for which we are searching for a solution f(x, y)=0.
    df_x : function
        Partial derivative of f(x, y) with respect to x.
    df_y : function
        Partial derivative of f(x, y) with respect to y.
    starting_point: list
        Initial guess for a solution f(x, y)=0.
    iterations: integer
        Maximum number of iterations of Newton's method.
    epsilon: number
        Stopping criteria.
    step: float
    ax: plt.axes
    point_radius: float
    last_point_radius: float
    point_color: np.color
    last_point_color: np.color

    Returns
    -------
    point: list
        Found point.
    '''
    point = starting_point
    if ax:
        add_point(ax, *point, function(point), point_color, point_radius)
    for _ in range(iterations):
        if function(point) > epsilon:
            gradient = np.array([df_x(point), df_y(point)], dtype='double')
            point -= step*gradient
            if ax:
                add_point(ax, *point, function(point), point_color, point_radius)
        else:
            break
    if ax:
        add_point(ax, *point, function(point), last_point_color, last_point_radius)
    return point


def newton_method(function, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y,
                  starting_point, step, iterations, epsilon,
                  ax=None, point_radius=0.5, last_point_radius=1,
                  point_color='red', last_point_color='blue'):
    '''Approximate solution of f(x, y)=0 by Newton's method.

    Parameters
    ----------
    function : function
        Function for which we are searching for a solution f(x, y)=0.
    df_x : function
        Partial derivative of f(x, y) with respect to x.
    df_y : function
        Partial derivative of f(x, y) with respect to y.
    df_x_df_x : function
        Partial derivative of df_x(x, y) with respect to x.
    df_y_df_y : function
        Partial derivative of df_y(x, y) with respect to y.
    df_x_df_y : function
        Partial derivative of df_y(x, y) with respect to x.
    starting_point: list
        Initial guess for a solution f(x, y)=0.
    iterations: integer
        Maximum number of iterations of Newton's method.
    epsilon: number
        Stopping criteria.
    step: float
    ax: plt.axes
    point_radius: float
    last_point_radius: float
    point_color: np.color
    last_point_color: np.color

    Returns
    -------
    point: list
        Found point.
    '''
    point = starting_point
    if ax:
        add_point(ax, *point, function(point), point_color, point_radius)
    for _ in range(iterations):
        if function(point) > epsilon:
            hessian = np.array([[df_x_df_x(point), df_x_df_y(point)],
                                [df_x_df_y(point), df_y_df_y(point)]])
            gradient = np.array([df_x(point), df_y(point)])
            hessian_inv = np.linalg.inv(hessian)
            d = hessian_inv.dot(gradient)
            point -= step*d
            if ax:
                add_point(ax, *point, function(point), point_color, point_radius)
        else:
            break
    if ax:
        add_point(ax, *point, function(point), last_point_color, last_point_radius)
    return point


def add_point(ax, x, y, z, color=None, radius=0.05):
    '''Add a point in a shape of ellipse to graph.'''
    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation = {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                     'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                     'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width=radius, height=radius*ratio, fc=color)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)


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

    parser = argparse.ArgumentParser(
        description=
        """Minimize Rosenbrock's function
        using Steepest Gradient Descent or Newton's method."""
        )

    parser.add_argument('x', metavar='X', type=float, nargs=1,
                        help='starting point (x)')
    parser.add_argument('y', metavar='Y', type=float, nargs=1,
                        help='starting point (y)')
    parser.add_argument('step', type=float, nargs=1,
                        help='value of step')
    parser.add_argument('iterations', type=int, nargs=1,
                        help='maximum number of iterations')
    parser.add_argument('epsilon', type=float, nargs=1,
                        help='stopping criteria')
    parser.add_argument('--method', action='store', default="steepest_descent",
                        help="""method to be used ("steepest_descent" or "newton")
                                (default: steepest_descent)""")
    parser.add_argument('--graph', action='store_true',
                        help='show the graph')
    args = parser.parse_args()

    starting_point = np.array([args.x[0], args.y[0]], dtype='float')
    step = args.step[0]
    iterations = args.iterations[0]
    epsilon = args.epsilon[0]

    if args.graph:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = np.linspace(-6, 6, 30)
        y = np.linspace(-6, 6, 30)
        X, Y = np.meshgrid(x, y)
        Z = f([X, Y])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', linewidth=0.3, alpha=0.8, edgecolor='k')
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if args.method == "steepest_descent":
            point = steepest_descent(f, df_x, df_y, starting_point, step, iterations, epsilon, ax)
        elif args.method == "newton":
            point = newton_method(f, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y, starting_point, step, iterations, epsilon, ax)
        else:
            print("Wrong name of the method!")
        minimum = f(point)
        print(f'f({point[0]}, {point[1]}) = {minimum})')
        plt.show()
    else:
        if args.method == "steepest_descent":
            point = steepest_descent(f, df_x, df_y, starting_point, step, iterations, epsilon)
        elif args.method == "newton":
            point = newton_method(f, df_x, df_y, df_x_df_x, df_y_df_y, df_x_df_y, starting_point, step, iterations, epsilon)
        else:
            print("Wrong name of the method!")
        minimum = f(point)
        print(f'f({point[0]}, {point[1]}) = {minimum})')
