import pybullet
import pybullet_data

# Helpers
def cross(v0, v1):
    return [ v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0] ]

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

def rotationInverse(q):
    scale = 1.0 / max(0.0001, np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]))

    return [ -q[0] * scale, -q[1] * scale, -q[2] * scale, q[3] * scale ]

class Sim:
    def __init__(
        self,
        xml_path,
        kp=0.25,
        kv=0.5,
        max_torque=10,
        g=-9.81,
    ):
        self.xml_path = xml_path
        self.g = g
        
        # Set up PyBullet Simulator
        pybullet.connect(pybullet.GUI)  # or p.DIRECT for non-graphical version
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        
        self.reset()

        # Set default camera a bit closer
        camInfo = pybullet.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        yaw = camInfo[8]
        pitch = camInfo[9]

        pybullet.resetDebugVisualizerCamera(1.0, yaw, pitch, pybullet.getBasePositionAndOrientation(self.model[1])[0])

        print("")
        print("Pupper body IDs:", self.model)
        numjoints = pybullet.getNumJoints(self.model[1])
        print("Number of joints in converted MJCF: ", numjoints)
        print("Joint Info: ")
        for i in range(numjoints):
            print(pybullet.getJointInfo(self.model[1], i))
        self.joint_indices = list(range(0, 24, 2))

        # Additional runtime params
        self.tipThresh = 0.2
        self.yawThresh = 0.3

    def reset(self):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, self.g)

        self.model = pybullet.loadMJCF(self.xml_path)

        #pybullet.resetBasePositionAndOrientation(self.model[1], [ 0, 0, 0.3 ], [ 0, 0, 0, 1 ])

    def step(self):
        # Follow camera
        camInfo = pybullet.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]

        vels = pybullet.getBaseVelocity(self.model[1])

        reward = vels[0][0]

        posOrient = pybullet.getBasePositionAndOrientation(self.model[1])

        upVec = rotateVec(posOrient[1], [ 0.0, 0.0, 1.0 ])
        forwardVec = rotateVec(posOrient[1], [ 1.0, 0.0, 0.0 ])

        if upVec[2] < self.tipThresh or forwardVec[0] < self.yawThresh:
            reward = -100.0

            self.reset()
        else:
            pybullet.resetDebugVisualizerCamera(distance, yaw, pitch, posOrient[0])

            pybullet.stepSimulation()

        return reward, vels
