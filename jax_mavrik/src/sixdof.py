import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, Optional
from jaxtyping import Float

from jax_mavrik.src.utils.jax_types import FloatScalar
from jax_mavrik.src.utils.mat_tools import euler_to_dcm
import diffrax
from jax import numpy as jnp
from jax import jit
from jax import lax


class RigidBody(NamedTuple):
    mass: Float
    inertia: Float

class SixDOFState(NamedTuple): 
    Ve: FloatScalar
    Xe: FloatScalar
    Vb: FloatScalar 
    Euler: FloatScalar
    pqr: FloatScalar
    ab: FloatScalar
    dotpqr: FloatScalar


class SixDOFDynamics:
    """
    Class to simulate 6 Degrees of Freedom (6DOF) dynamics using Euler angles, following
    the behavior of the MathWorks 6DOF block.
    """

    def __init__(self, rigid_body: RigidBody, method: str = 'RK4', fixed_step_size: float = 0.01):
        """
        Initialize the 6DOF dynamics simulator.

        Args:
            rigid_body (RigidBody): Rigid body object containing mass and inertia.
            method (str): Integration method ("RK4", "Euler", or "diffrax"). 
            fixd_step_size (float): Fixed step size for the simulation.
        """
        self.rigid_body = rigid_body 
        self.method = method 
        self.fixed_step_size = fixed_step_size
        
    def _six_dof_dynamics(self, state, Fxyz, Mxyz):
        """
        Defines the 6DOF dynamics equations of motion based on Newton's and Euler's equations.
        
        Args:
            state (numpy.ndarray): Current state vector [u, v, w, phi, theta, psi, p, q, r].
            Fxyz (array-like): Forces acting on the rigid body (Fx, Fy, Fz).
            Mxyz (array-like): Moments acting on the rigid body (Mx, My, Mz).
        
        Returns:
            numpy.ndarray: Derivative of the state vector.
        """
         
        array, cos, sin, tan, concatenate, linalg, cross = np.asarray, np.cos, np.sin, np.tan, np.concatenate, np.linalg, np.cross
        if isinstance(state, jnp.ndarray):
            array, cos, sin, tan, concatenate, linalg, cross = jnp.asarray, jnp.cos, jnp.sin, jnp.tan, jnp.concatenate, jnp.linalg, jnp.cross
          
        _, _, _, _, _, _, u, v, w, phi, theta, psi, p, q, r, _, _, _, _, _, _ = state

        # Convert body-frame forces (Fxyz) to NED-frame forces
        L_EB = array(euler_to_dcm(phi, theta, psi)).T  # Transpose to convert body to NED

        # Position derivatives in the NED frame (convert body velocities to NED frame)
        dXe, dYe, dZe = L_EB @ array([u, v, w])
        
        # Translational acceleration in the body frame
        du = Fxyz[0] / self.rigid_body.mass + r * v - q * w
        dv = Fxyz[1] / self.rigid_body.mass + p * w - r * u
        dw = Fxyz[2] / self.rigid_body.mass + q * u - p * v

        dVXe, dVYe, dVZe = L_EB @ array([du, dv, dw]) 
      
        # Rotational motion (Euler's equations in the body frame)
        angular_velocity = array([p, q, r])
        I = self.rigid_body.inertia
        I_inv = linalg.inv(I)
        Mxyz = array(Mxyz)
        dp, dq, dr = I_inv @ (Mxyz - cross(angular_velocity, I @ angular_velocity))
         
        # Euler angles rates
        dphi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        epsilon = 1e-6
        dpsi = (q * sin(phi) + r * cos(phi)) / (cos(theta) + epsilon)
         
        # Return the state offsets
        return concatenate([array([dVXe, dVYe, dVZe]), array([dXe, dYe, dZe]), array([du, dv, dw]), array([dphi, dtheta, dpsi]), array([dp, dq, dr]), array([du, dv, dw]), array([dp, dq, dr])])
    
    
 
    def run_simulation(self, initial_state: SixDOFState, forces: FloatScalar, moments: FloatScalar, t0=0.0, t1=0.01):
        """
        Run the 6DOF dynamics simulation.
        
        Args:
            initial_state (State): Initial state object.
            forces (array-like): External forces in the body frame (Fx, Fy, Fz).
            moments (array-like): External moments in the body frame (Mx, My, Mz).
            t0 (float): Initial time of the simulation.
            t1 (float): Final time of the simulation.
            num_points (int): Number of points for evaluation.
            method (str): Integration method ("RK4" or "diffrax").
        
        Returns:
            dict: A dictionary containing time and state history.
        """
        initial_state_vector = jnp.concatenate([
            initial_state.Ve,
            initial_state.Xe,
            initial_state.Vb, 
            initial_state.Euler,
            initial_state.pqr,
            jnp.zeros(6)  # Add zeros for dotpqr and ab
        ])
        forces = jnp.asarray(forces)
        moments = jnp.asarray(moments)

        num_points = jnp.ceil((t1 - t0) / self.fixed_step_size).astype(int)
        times = jnp.linspace(t0, t1, num_points)
        if self.method.lower() == "rk4":
            def rk4_step(state, forces_moments):
                state = state.at[-6:].set(jnp.zeros(6))
                forces, moments = forces_moments
                forces, moments = jnp.asarray(forces), jnp.asarray(moments)
                k1 = self.fixed_step_size * self._six_dof_dynamics(state, forces, moments)
                k2 = self.fixed_step_size * self._six_dof_dynamics(state + k1/2, forces, moments)
                k3 = self.fixed_step_size * self._six_dof_dynamics(state + k2/2, forces, moments)
                k4 = self.fixed_step_size * self._six_dof_dynamics(state + k3, forces, moments)
                new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
                return new_state, new_state

            forces_moments = (jnp.tile(forces, (num_points, 1)), jnp.tile(moments, (num_points, 1)))
            _, states = lax.scan(rk4_step, initial_state_vector, forces_moments)

        elif self.method.lower() == "euler":
            def euler_step(state, forces_moments):
                state = state.at[-6:].set(jnp.zeros(6))
                forces, moments = forces_moments
                forces, moments = jnp.asarray(forces), jnp.asarray(moments)
                new_state = state + self.fixed_step_size * self._six_dof_dynamics(state, forces, moments)
                return new_state, new_state

            forces_moments = (jnp.tile(forces, (num_points, 1)), jnp.tile(moments, (num_points, 1)))
            _, states = lax.scan(euler_step, initial_state_vector, forces_moments)
        elif self.method.lower() == "diffrax":
            ### diffrax has some resolved issues:
            ##### The first state output will remain the same as the initial state
            ##### Recommend using RK4 or Euler methods for now
            def dynamics(t, state, args):
                state = state.at[-6:].set(jnp.zeros(6))
                forces, moments = jnp.asarray(args[0]), jnp.asarray(args[1])
                return jnp.array(self._six_dof_dynamics(state, forces, moments))

            solver = diffrax.Tsit5()
            term = diffrax.ODETerm(dynamics)
            saveat = diffrax.SaveAt(ts=times)
            sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=self.fixed_step_size, y0=initial_state_vector, args=(forces, moments), saveat=saveat)
            states = jnp.array(sol.ys)

        return {"time": times, "states": states} # Return time and state history

if __name__ == "__main__":
    # Define constants and initial state
    mass = 10.0
    inertia = np.diag([0.5, 0.5, 0.8])
    forces = [0., 0., -mass * 9.81]  # Gravity in the body frame
    moments = [0., 0., 0.]  # No initial moments
    t0, t1 = 0.0, 30.0
    initial_state = SixDOFState(
        Ve=np.array([0., 0., 0.]),
        Xe=np.array([0., 0., 0.]),
        Vb=np.array([30., 0., 0.]),
        Euler=np.array([0., 0., 0.]),
        pqr=np.array([0., 0., 0.]),
        ab=np.array([0., 0., 0.]),
        dotpqr=np.array([0., 0., 0.])
    )

    rigid_body = RigidBody(mass=mass, inertia=np.array(inertia))

    plt.figure()
    
    ## Testing RK4 method
    dynamics = SixDOFDynamics(rigid_body, method="RK4", fixed_step_size=0.01)
    results_rk4 = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    # Plot results for RK4 method (position over time as an example)
    time_rk4 = results_rk4["time"]
    position_rk4 = results_rk4["states"][:, 3:6]  # x, y, z positions
    plt.plot(time_rk4, position_rk4[:, 0], label="X Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 1], label="Y Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 2], label="Z Position (RK4)")

    ## Testing Euler method
    dynamics = SixDOFDynamics(rigid_body, method="euler", fixed_step_size=0.01)
    results_euler = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    time_euler = results_euler["time"]
    position_euler = results_euler["states"][:, 3:6]  # x, y, z positions
    plt.plot(time_euler, position_euler[:, 0], label="X Position (Euler)")
    plt.plot(time_euler, position_euler[:, 1], label="Y Position (Euler)")
    plt.plot(time_euler, position_euler[:, 2], label="Z Position (Euler)")

    ## Testing diffrax method
    dynamics = SixDOFDynamics(rigid_body, method="diffrax", fixed_step_size=0.01)
    results_diffrax = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    time_diffrax = results_diffrax["time"]
    position_diffrax = results_diffrax["states"][:, 3:6]  # x, y, z positions
    plt.plot(time_diffrax, position_diffrax[:, 0], label="X Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 1], label="Y Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 2], label="Z Position (diffrax)")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (All Methods)")
    plt.legend()
    plt.show()
