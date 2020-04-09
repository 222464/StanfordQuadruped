from pupper.Config import ServoParams, PWMParams

import pyogmaneo
from pyogmaneo import Int3

import numpy as np

ANGLE_RESOLUTION = 16
IMU_RESOLUTION = 16
IMU_SQUASH_SCALE = 1.0

class TrainingInterface:
    def __init__(self, imu=None):
        self.imu = imu

        self.cs = pyogmaneo.ComputeSystem()

        lds = []

        for i in range(3):
            ld = pyogmaneo.LayerDesc()
            ld.hiddenSize = Int3(4, 4, 16)

            ld.ffRadius = 4
            ld.pRadius = 4
            ld.aRadius = 4

            ld.ticksPerUpdate = 2
            ld.temporalHorizon = 2

            lds.append(ld)

        input_sizes = [ Int3(4, 3, ANGLE_RESOLUTION) ]
        input_types = [ pyogmaneo.inputTypeAction ]

        if self.imu is not None:
            input_sizes.append(Int3(3, 2, IMU_RESOLUTION))
            input_types.append(pyogmaneo.inputTypeNone)

        self.h = pyogmaneo.Hierarchy(self.cs, input_sizes, input_types, lds)

        self.reward = 0.0

        self.average_error = 0.0
        self.average_error_decay = 0.999

        self.num_samples = 0

    def set_actuator_positions(self, joint_angles):
        joint_angles_raveled = joint_angles.ravel()

        angle_SDR = [ int((min(1.0, max(-1.0, joint_angles_raveled[i] / (0.5 * np.pi))) * 0.5 + 0.5) * (ANGLE_RESOLUTION - 1) + 0.5) for i in range(len(joint_angles_raveled)) ]

        if self.imu is None:
            # Compare predictions
            error = 0.0

            predictions = list(self.h.getPredictionCs(0))

            for i in range(len(predictions)):
                dist = angle_SDR[i] - predictions[i]

                error += dist * dist

            error /= len(predictions)

            self.average_error = self.average_error_decay * self.average_error + (1.0 - self.average_error_decay) * error

            # Update agent
            self.h.step(self.cs, [ angle_SDR ], True, 1.0)
        else:
            pass # TODO: Implement?

        #print(angle_SDR)

        self.num_samples += 1

        if self.num_samples % 100 == 0:
            print("Sample " + str(self.num_samples) + ", error: " + str(self.average_error))
    
    def save(self, fileName):
        self.h.save(fileName)
        