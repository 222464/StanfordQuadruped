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
from pyogmaneo import Int3

ANGLE_RESOLUTION = 16
IMU_RESOLUTION = 16
IMU_SQUASH_SCALE = 1.0

def main(use_imu=False, default_velocity=np.zeros(2), default_yaw_rate=0.0, lock_frame_rate=True):
    # Create config
    config = Configuration()
    config.z_clearance = 0.05
    sim = Sim()
    hardware_interface = HardwareInterface(sim.model, sim.joint_indices)

    # Create imu handle
    if use_imu:
        imu = IMU()

    # Load hierarchy
    cs = pyogmaneo.ComputeSystem()

    # lds = []

    # for i in range(5):
    #     ld = pyogmaneo.LayerDesc()
    #     ld.hiddenSize = Int3(4, 4, 16)

    #     ld.ffRadius = 4
    #     ld.pRadius = 4
    #     ld.aRadius = 4

    #     ld.ticksPerUpdate = 2
    #     ld.temporalHorizon = 2

    #     lds.append(ld)

    # input_sizes = [ Int3(4, 3, ANGLE_RESOLUTION) ]
    # input_types = [ pyogmaneo.inputTypeAction ]

    # input_sizes.append(Int3(3, 2, IMU_RESOLUTION))
    # input_types.append(pyogmaneo.inputTypeNone)

    #h = pyogmaneo.Hierarchy(cs, input_sizes, input_types, lds)

    h = pyogmaneo.Hierarchy("pupper.ohr")

    angles = 12 * [ 0.0 ]

    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)

    # Sim seconds per sim step
    sim_steps_per_sim_second = 240
    sim_dt = 1.0 / sim_steps_per_sim_second

    start_sim_time = time.time()

    sim_time_elapsed = 0.0

    reward = 0.0
    control_reward_accum = 0.0
    control_reward_accum_steps = 0
    vels = ( [ 0, 0, 0 ], [ 0, 0, 0 ] )
    steps = 0

    while True:
        start_step_time = time.time()

        sim_time_elapsed += sim_dt
        
        if sim_time_elapsed > config.dt:
            sim_time_elapsed = sim_time_elapsed % config.dt

            imu_vals = list(vels[0]) + list(vels[1])

            imu_SDR = []

            for i in range(len(imu_vals)):
                imu_SDR.append(int((np.tanh(imu_vals[i] * IMU_SQUASH_SCALE) * 0.5 + 0.5) * (IMU_RESOLUTION - 1) + 0.5))

            h.step(cs, [ h.getPredictionCs(0), imu_SDR ], True, control_reward_accum / max(1, control_reward_accum_steps))
            
            control_reward_accum = 0.0
            control_reward_accum_steps = 0

            joint_angles = np.zeros((3, 4))

            motor_index = 0

            for segment_index in range(3):
                for leg_index in range(4):
                    target_angle = (h.getPredictionCs(0)[motor_index] / float(ANGLE_RESOLUTION - 1) * 2.0 - 1.0) * (0.5 * np.pi)
                    
                    delta = 0.3 * (target_angle - angles[motor_index])

                    max_delta = 0.04

                    if abs(delta) > max_delta:
                        delta = max_delta if delta > 0.0 else -max_delta

                    angles[motor_index] += delta

                    joint_angles[segment_index, leg_index] = angles[motor_index]

                    motor_index += 1

            # Update the pwm widths going to the servos
            hardware_interface.set_actuator_postions(joint_angles)

        # Simulate physics for 1/240 seconds (the default timestep)
        reward, vels = sim.step()

        control_reward_accum += reward
        control_reward_accum_steps += 1

        if steps % 10000 == 9999:
            print("Saving...")
            h.save("pupper_rltrained.ohr")

        steps += 1

        # Performance testing
        step_elapsed = time.time() - start_step_time

        # Keep framerate
        if lock_frame_rate:
            time.sleep(max(0, sim_dt - step_elapsed))

if __name__ == "__main__":
    main(default_velocity=np.array([0.5, 0]), lock_frame_rate=False)
