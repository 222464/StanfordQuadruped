import pyogmaneo
from pyogmaneo import Int3

import numpy as np

def rotateVec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [ v[0] + uv[0] + uuv[0],
        v[1] + uv[1] + uuv[1],
        v[2] + uv[2] + uuv[2] ]

def mutate(x, rate, oneHotSize):
    z = copy(x)

    indices = np.where(np.random.rand(len(z)) < rate)

    randSDR = np.random.randint(0, oneHotSize, size=(len(x)))

    z[indices] = randSDR[indices]

    return z

ANGLE_RESOLUTION = 16
IMU_RESOLUTION = 16

class TrainingInterface:
    def __init__(self):
        pyogmaneo.ComputeSystem.setNumThreads(4)
        self.cs = pyogmaneo.ComputeSystem()

        lds = []

        for i in range(4):
            ld = pyogmaneo.LayerDesc()
            ld.hiddenSize = Int3(4, 4, 16)

            ld.ffRadius = 4
            ld.pRadius = 4
            ld.aRadius = 4

            ld.ticksPerUpdate = 2
            ld.temporalHorizon = 4

            lds.append(ld)

        input_sizes = [ Int3(4, 3, ANGLE_RESOLUTION) ]
        input_types = [ pyogmaneo.inputTypeAction ]

        input_sizes.append(Int3(3, 2, IMU_RESOLUTION))
        input_types.append(pyogmaneo.inputTypeNone)

        self.h = pyogmaneo.Hierarchy(self.cs, input_sizes, input_types, lds)

        self.reward = 1.0

        self.average_error = 0.0
        self.average_error_decay = 0.999

        self.num_samples = 0

    def set_reward(self, reward):
        self.reward = reward

    def set_actuator_positions(self, joint_angles):
        joint_angles_raveled = joint_angles.ravel()

        # Mutate
        #joint_angles_mut = joint_angles_raveled + np.random.randn(joint_angles_raveled.shape[0]) * 0.1

        #noise_error = np.sum(np.square(joint_angles_raveled - joint_angles_mut))
        
        angle_SDR = [ int((min(1.0, max(-1.0, joint_angles_raveled[i] / (0.5 * np.pi))) * 0.5 + 0.5) * (ANGLE_RESOLUTION - 1) + 0.5) for i in range(len(joint_angles_raveled)) ]

        # Compare predictions
        error = 0.0

        predictions = list(self.h.getPredictionCs(0))

        for i in range(len(predictions)):
            dist = angle_SDR[i] - predictions[i]

            error += dist * dist

        error /= len(predictions)

        self.average_error = self.average_error_decay * self.average_error + (1.0 - self.average_error_decay) * error

        # Update agent
        self.h.step(self.cs, [ angle_SDR, [ np.random.randint(0, IMU_RESOLUTION) for i in range(6) ] ], True, self.reward) # Constant positive reward encourages prediction of angles

        self.num_samples += 1

        # Give updates on training
        if self.num_samples % 1000 == 0:
            print("Sample " + str(self.num_samples) + ", average error: " + str(self.average_error))

        # Save every now and then
        if self.num_samples % 10000 == 0:
            print("Saving...")

            self.h.save("pupper.ohr")

            print("Saved.")
        