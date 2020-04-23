import pybullet as p
import pybullet_data
import time
import numpy as np
from copy import copy

from sim.IMU import IMU
from sim.Sim import Sim
from common.Controller import Controller
from common.Command import Command
from common.State import State
from sim.HardwareInterface import HardwareInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from common.State import State, BehaviorState

import pyogmaneo
from pyogmaneo import Int3

import evdev
from evdev import list_devices, ecodes

ANGLE_RESOLUTION = 16
COMMAND_RESOLUTION = 16
IMU_RESOLUTION = 16
IMU_SQUASH_SCALE = 1.0

ls = [ 0.0, 0.0 ]
rs = [ 0.0, 0.0 ]

def mutate(x, rate, oneHotSize):
    z = copy(x)

    indices = np.where(np.random.rand(len(z)) < rate)

    randSDR = np.random.randint(0, oneHotSize, size=(len(x)))

    z[indices] = randSDR[indices]

    return z

def main(use_imu=False, default_velocity=np.zeros(2), default_yaw_rate=0.0, lock_frame_rate=True):
    device = evdev.InputDevice(list_devices()[0])
    print(device)

    # Create config
    sim = Sim()
    hardware_interface = HardwareInterface(sim.model, sim.joint_indices)

    # Create imu handle
    if use_imu:
        imu = IMU()

    # Load hierarchy
    pyogmaneo.ComputeSystem.setNumThreads(4)
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
    #h = pyogmaneo.Hierarchy("pupper_rltrained.ohr")

    angles = 12 * [ 0.0 ]

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

    actions = list(h.getPredictionCs(0))

    # Create config
    config = Configuration()
    config.z_clearance = 0.05

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()
    state.behavior_state = BehaviorState.TROT

    octaves = 3
    smooth_chain = octaves * [ np.array([ 0.0, 0.0, 0.0 ]) ]
    smooth_factor = 0.005
    smooth_scale = 4.0
    max_speed = 0.5
    max_yaw_rate = 1.0

    offsets = np.array([ -0.12295051, 0.12295051, -0.12295051, 0.12295051, 0.77062617, 0.77062617,
        0.77062617, 0.77062617, -0.845151, -0.845151, -0.845151, -0.845151 ])

    while True:
        start_step_time = time.time()

        sim_time_elapsed += sim_dt
        
        if sim_time_elapsed > config.dt:
            
            try:
                for event in device.read():
                    if event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_X:
                            ls[0] = event.value / 32767.0
                        elif event.code == ecodes.ABS_Y:
                            ls[1] = event.value / 32767.0
                        elif event.code == ecodes.ABS_RX:
                            rs[0] = event.value / 32767.0
                        elif event.code == ecodes.ABS_RY:
                            rs[1] = event.value / 32767.0
            except:
                pass

            sim_time_elapsed = sim_time_elapsed % config.dt

            imu_vals = list(vels[0]) + list(vels[1])

            imu_SDR = []

            for i in range(len(imu_vals)):
                imu_SDR.append(IMU_RESOLUTION // 2)#int((np.tanh(imu_vals[i] * IMU_SQUASH_SCALE) * 0.5 + 0.5) * (IMU_RESOLUTION - 1) + 0.5))

            # Smoothed noise
            smooth_chain[0] += smooth_factor * (np.random.randn(3) * smooth_scale - smooth_chain[0])

            for i in range(1, octaves):
                smooth_chain[i] += smooth_factor * (smooth_chain[i - 1] - smooth_chain[i])

            smoothed_result = np.minimum(1.0, np.maximum(-1.0, smooth_chain[-1]))

            #direction = smoothed_result
            #direction = np.array([ 1.0, 0.0, 0.0 ])
            direction = np.array([ -ls[1], -ls[0], -rs[0] ])
            
            command_SDR = [ int((direction[i] * 0.5 + 0.5) * (COMMAND_RESOLUTION - 1) + 0.5) for i in range(3) ]
            
            h.step(cs, [ actions, command_SDR, imu_SDR ], False, control_reward_accum / max(1, control_reward_accum_steps))
  
            actions = list(h.getPredictionCs(0))

            #actions = mutate(np.array(actions), 0.01, h.getInputSize(0).z).tolist()

            control_reward_accum = 0.0
            control_reward_accum_steps = 0

            joint_angles = np.zeros((3, 4))

            motor_index = 0

            for segment_index in range(3):
                for leg_index in range(4):
                    target_angle = (actions[motor_index] / float(ANGLE_RESOLUTION - 1) * 2.0 - 1.0) * (0.25 * np.pi) + offsets[motor_index]
                    
                    delta = 0.5 * (target_angle - angles[motor_index])

                    max_delta = 0.05

                    if abs(delta) > max_delta:
                        delta = max_delta if delta > 0.0 else -max_delta

                    angles[motor_index] += delta

                    joint_angles[segment_index, leg_index] = angles[motor_index]

                    motor_index += 1

            command = Command()

            # Go forward at max speed
            command.horizontal_velocity = direction[0 : 2] * max_speed
            command.yaw_rate = direction[2] * max_yaw_rate

            quat_orientation = (
                np.array([1, 0, 0, 0])
            )
            state.quat_orientation = quat_orientation

            # Step the controller forward by dt
            controller.run(state, command)
            
            #joint_angles = copy(state.joint_angles)

            # Update the pwm widths going to the servos
            hardware_interface.set_actuator_postions(joint_angles)

        # Simulate physics for 1/240 seconds (the default timestep)
        reward, vels = sim.step()

        control_reward_accum += reward
        control_reward_accum_steps += 1

        if steps % 50000 == 49999:
            print("Saving...")
            h.save("pupper_rltrained.ohr")

        steps += 1

        # Performance testing
        step_elapsed = time.time() - start_step_time

        # Keep framerate
        if lock_frame_rate:
            time.sleep(max(0, sim_dt - step_elapsed))

    pygame.quit()

if __name__ == "__main__":
    main(default_velocity=np.array([0.5, 0]), lock_frame_rate=True)
