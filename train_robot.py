import numpy as np
import time
from src.Controller import Controller
from src.State import State, BehaviorState
from src.Command import Command
from pupper.TrainingInterface import TrainingInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

def main(use_imu=False):
    """Main program
    """

    # Create config
    config = Configuration()
    training_interface = TrainingInterface()

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()

    # Behavior to learn
    state.behavior_state = BehaviorState.TROT

    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)

    amplitude = 0.0
    amplitude_vel = 0.0
    angle = 0.0
    angle_vel = 0.0
    yaw = 0.0
    yaw_vel = 0.0

    while True:
        # Parse the udp joystick commands and then update the robot controller's parameters
        command = Command()

        amplitude_accel = np.random.randn() * 3.0
        amplitude_vel += amplitude_accel * config.dt - 0.2 * amplitude_vel * config.dt
        amplitude += amplitude_vel * config.dt

        angle_accel = np.random.randn() * 3.0
        angle_vel += angle_accel * config.dt - 0.2 * angle_vel * config.dt
        angle += angle_vel * config.dt

        yaw_accel = np.random.randn() * 3.0
        yaw_vel += yaw_accel * config.dt - 0.2 * yaw_vel * config.dt
        yaw += yaw_vel * config.dt

        #print(str(amplitude) + " " + str(angle) + " " + str(yaw))

        # Go forward at max speed
        command.horizontal_velocity = np.array([ np.cos(angle) * np.sin(amplitude), np.sin(angle) * np.sin(amplitude) ]) * 0.5
        command.yaw_rate = np.sin(yaw) * 0.5

        quat_orientation = (
            np.array([1, 0, 0, 0])
        )
        state.quat_orientation = quat_orientation

        # Step the controller forward by dt
        controller.run(state, command)

        training_interface.set_direction(np.array([ command.horizontal_velocity[0], command.horizontal_velocity[1], command.yaw_rate ]))

        # Update the agent with the angles
        training_interface.set_actuator_positions(state.joint_angles)


main()
