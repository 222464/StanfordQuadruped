import pybullet as p
import pybullet_data
import time
import numpy as np

from sim.IMU import IMU
from sim.Sim import Sim
from common.Controller import Controller
from common.Command import Command
from common.State import State
from sim.HardwareInterface import HardwareInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

import pyogmaneo


def main(use_imu=False, default_velocity=np.zeros(2), default_yaw_rate=0.0, lock_frame_rate=True):
    # Create config
    config = Configuration()
    config.z_clearance = 0.05
    sim = Sim(xml_path="sim/pupper_pybullet_out.xml")
    hardware_interface = HardwareInterface(sim.model, sim.joint_indices)

    # Create imu handle
    if use_imu:
        imu = IMU()

    # Load hierarchy
    cs = pyogmaneo.ComputeSystem()

    h = pyogmaneo.Hierarchy("pupper.ohr")

    angles = 12 * [ 0.0 ]

    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)

    # Run the simulation
    timesteps = 240 * 60 * 10  # simulate for a max of 10 minutes

    # Sim seconds per sim step
    sim_steps_per_sim_second = 240
    sim_dt = 1.0 / sim_steps_per_sim_second
    last_control_update = 0

    start_sim_time = time.time()

    for steps in range(timesteps):
        start_step_time = time.time()

        sim_time_elapsed = sim_dt * steps
        if sim_time_elapsed - last_control_update > config.dt:
            last_control_update = sim_time_elapsed

            # Get IMU measurement if enabled
            quat_orientation = (
                imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
            )
            
            ANGLE_RES = h.getInputSize(0).z

            h.step(cs, [ h.getPredictionCs(0) ], False, 1.0)

            joint_angles = np.zeros((3, 4))

            motor_index = 0

            for segment_index in range(3):
                for leg_index in range(4):
                    target_angle = (h.getPredictionCs(0)[motor_index] / float(ANGLE_RES - 1) * 2.0 - 1.0) * (0.5 * np.pi)
                    
                    angles[motor_index] += 0.3 * (target_angle - angles[motor_index])

                    joint_angles[segment_index, leg_index] = angles[motor_index]

                    motor_index += 1

            # Update the pwm widths going to the servos
            hardware_interface.set_actuator_postions(joint_angles)

        # Simulate physics for 1/240 seconds (the default timestep)
        sim.step()

        # Performance testing
        step_elapsed = time.time() - start_step_time

        if (steps % 60) == 0:
            print(
                "Sim seconds elapsed: {}, Real seconds elapsed: {}".format(
                    round(sim_time_elapsed, 3), round(time.time() - start_sim_time, 3)
                )
            )
            # print("Average steps per second: {0}, elapsed: {1}, i:{2}".format(steps / elapsed, elapsed, i))

        # Keep framerate
        if lock_frame_rate:
            time.sleep(max(0, sim_dt - step_elapsed))

if __name__ == "__main__":
    main(default_velocity=np.array([0.5, 0]))
