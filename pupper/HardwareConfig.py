"""
Per-robot configuration file that is particular to each individual robot, not just the type of robot.
"""
import numpy as np


MICROS_PER_RAD = 11.333 * 180.0 / np.pi  # Must be calibrated
NEUTRAL_ANGLE_DEGREES = np.array(
    [[ -7.0, -4.0, -4.0, -3.0 ], [ 43.0, 46.0, 42.0, 42.0 ], [ -34.0, -35.0, -34.0, -34.0 ]]
)

PS4_COLOR = {"red": 0, "blue": 0, "green": 255}
PS4_DEACTIVATED_COLOR = {"red": 0, "blue": 0, "green": 50}