import numpy as np
import time
from common.Controller import Controller
from common.State import State, BehaviorState
from common.Command import Command
from pupper.TrainingInterface import TrainingInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

def main(use_imu=False):
    """Main program
    """

    # Create config
    config = Configuration()
    config.z_clearance = 0.05
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

    octaves = 3
    smooth_chain = octaves * [ np.array([ 0.0, 0.0, 0.0 ]) ]
    smooth_factor = 0.01
    smooth_scale = 4.0
    max_speed = 0.5
    max_yaw_rate = 4.0

    while True:
        # Parse the udp joystick commands and then update the robot controller's parameters
        command = Command()

        # Smoothed noise
        smooth_chain[0] += smooth_factor * (np.random.randn(3) * smooth_scale - smooth_chain[0])

        for i in range(1, octaves):
            smooth_chain[i] += smooth_factor * (smooth_chain[i - 1] - smooth_chain[i])

        smoothed_result = np.minimum(1.0, np.maximum(-1.0, smooth_chain[-1]))
        
        #direction = np.array([ 1.0, 0.0, 0.0 ])
        direction = smoothed_result

        training_interface.set_reward(1.0)
        training_interface.set_direction(direction)

        # Go forward at max speed
        command.horizontal_velocity = direction[0 : 2] * max_speed
        command.yaw_rate = direction[2] * max_yaw_rate

        quat_orientation = (
            np.array([1, 0, 0, 0])
        )
        state.quat_orientation = quat_orientation

        # Step the controller forward by dt
        controller.run(state, command)

        # Update the agent with the angles
        training_interface.set_actuator_positions(state.joint_angles)


main()