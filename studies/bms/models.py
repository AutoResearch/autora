from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.utils import lorenz


class Model:

    def __init__(self):
        self.t = np.zeros(shape=(1, 1))
        self.y = np.zeros(shape=(1, 1))
        self.dim = 0

    def visualize_model(self):
        if self.dim == 0:
            raise Warning("Model Uninitialized")
        elif self.dim == 1:
            plt.figure(figsize=(8, 4))
            plt.xlabel("Time")
            plt.ylabel("")
            for i in range(self.y.shape[0]):
                plt.plot(self.t, self.y[i,:])
            plt.show()
        elif self.dim == 2:
            plt.figure(figsize=(8, 4))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(self.y[0], self.y[1])
            plt.show()
        elif self.dim == 3:
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(self.y[0, :], self.y[1, :], self.y[2, :], 'k')
            ax1.set(xlabel='$x_0$', ylabel='$x_1$',
                    zlabel='$x_2$')
            plt.show()


class Model1D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 1


class Model2D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 2


class Model3D(Model):

    def __init__(self):
        Model.__init__(self)
        self.dim = 3


class BifurcationModel(Model2D):

    def __init__(self, x0=0., y0=1., dt=0.002, t_max=10., a=1., b=-2.5):
        Model2D.__init__(self)
        x_init = (x0, y0)  # initial conditions
        # Get timesteps
        t = np.linspace(0, t_max, int(t_max / dt))
        self.t = t

        def int_model_sim(x_init, t):
            x_dot = x_init[1] - a * x_init[0] ** 2
            y_dot = b * x_init[0]
            return x_dot, y_dot

        self.y = odeint(int_model_sim, x_init, t).T


class CliffordModel(Model2D):

    def __init__(self, x0=0., y0=0., dt=0.02, t_max=10., a=1., b=-2.5, c=-1., d=.5):
        Model2D.__init__(self)
        x_init = (x0, y0)  # initial conditions
        # Get timesteps
        t = np.linspace(0, t_max, int(t_max / dt))
        self.t = t

        def int_model_sim(x_init, t):
            x_dot = np.sin(a * x_init[1]) + c * np.cos(a * x_init[0])
            y_dot = np.sin(b * x_init[0]) + d * np.cos(b * x_init[1])
            return x_dot, y_dot

        self.y = odeint(int_model_sim, x_init, t).T


class DuffingModel(Model2D):

    def __init__(self, x0=0., y0=0., dt=0.02, t_max=10., a=.35, b=.3, w=np.pi):
        Model2D.__init__(self)
        x_init = (x0, y0)  # initial conditions
        # Get timesteps
        t = np.linspace(0, t_max, int(t_max / dt))
        self.t = t

        def int_model_sim(x_init, t):
            x_dot = x_init[1]
            y_dot = x_init[0] - x_init[0] ** 3 - a * x_init[1] + b * np.cos(w * t)
            return x_dot, y_dot

        self.y = odeint(int_model_sim, x_init, t).T


class LorenzModel(Model3D):

    def __init__(self):
        Model3D.__init__(self)
        # Initialize integrator keywords for solve_ivp to replicate the odeint defaults
        integrator_keywords = {}
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['method'] = 'LSODA'
        integrator_keywords['atol'] = 1e-12
        # Generate measurement data
        dt = .002
        t = np.arange(0, 10, dt)
        self.t = t
        x_init = [-8, 8, 27]
        t_span = (t[0], t[-1])
        self.y = solve_ivp(lorenz, t_span, x_init, t_eval=t, **integrator_keywords).y


class PendulumModel(Model1D):

    # Input constants
    m: float  # mass (kg)
    L: float  # length (m)
    b: float  # damping value (kg/m^2-s)
    #  g: float  # gravity (m/s^2)
    delta_t: float  # time step size (seconds)
    t_max: float  # max sim time (seconds)
    theta1_0: float  # initial angle (radians)
    theta2_0: float  # initial angular velocity (rad/s)

    def __init__(self, m=1., L=1., b=0.5, delta_t=0.02, t_max=10., theta1_0=np.pi/10, theta2_0=0., noise=0.):
        Model1D.__init__(self)
        self.g = 9.81
        theta_init = (theta1_0, theta2_0)  # initial conditions
        # Get timesteps
        t = np.linspace(0, t_max, int(t_max / delta_t))
        self.t = t

        def int_pendulum_sim(theta_init, t):
            theta_dot_1 = theta_init[1]
            theta_dot_2 = -b / m * theta_init[1] - self.g / L * np.sin(theta_init[0])
            return theta_dot_1, theta_dot_2

        self.y = odeint(int_pendulum_sim, theta_init, t). T
        self.y_hat = self.y + np.random.normal(loc=np.mean(self.y), scale=noise, size=self.y.shape)
