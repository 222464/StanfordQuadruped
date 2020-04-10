import panda3d
from panda3d.core import loadPrcFileData
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import AmbientLight, DirectionalLight, LightAttrib
from panda3d.core import NodePath
from panda3d.core import LVector3, LVector2
from panda3d.core import Vec3, Vec2, Quat
from panda3d.core import AntialiasAttrib
from panda3d.core import GeoMipTerrain
from panda3d.core import Texture
from panda3d.core import PNMImage

from direct.interval.IntervalGlobal import *  # Needed to use Intervals
from direct.gui.DirectGui import *

import blend2bam

import numpy as np

import pyogmaneo
from pyogmaneo import Int3

from copy import copy

import sys

dt = 0.01

class PupperDisplay(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.title = OnscreenText(text="Pupper Display",
                                  parent=base.a2dBottomCenter,
                                  fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.5),
                                  pos=(0, 0.1), scale=0.1)

        #base.disableMouse()

        self.accept("q", lambda: sys.exit())
        
        render.setShaderAuto()
        render.setAntialias(AntialiasAttrib.MMultisample)

        self.loadHierarchy()
        self.loadModels()
        self.setupLights()

        mainTask = taskMgr.add(self.mainLoop, 'mainLoopTask')

    def loadHierarchy(self):
        self.cs = pyogmaneo.ComputeSystem()
        self.h = pyogmaneo.Hierarchy("pupper.ohr")

    def loadModels(self):
        self.limbModel = loader.loadModel("models/limb/limb.bam")

        hw = 10.0
        hh = 20.0

        height = 27.0

        legStarts = [
            Vec3(hw, hh, height),
            Vec3(-hw, hh, height),
            Vec3(hw, -hh, height),
            Vec3(-hw, -hh, height)
        ]

        self.limbs = []

        self.angles = 12 * [ 0.0 ]
        self.rest_rots = []

        for i in range(4):
            femur = copy(self.limbModel)
            femur.setPos(legStarts[i])
            femur.reparentTo(render)

            self.limbs.append(femur)

            q = Quat()
            q.setFromAxisAngleRad(np.pi * 0.75, Vec3(1.0, 0.0, 0.0))

            self.rest_rots.append(q)

            tibia = copy(self.limbModel)
            tibia.setPos(Vec3(0.0, 0.0, 24.0))
            tibia.reparentTo(femur)

            self.limbs.append(tibia)

            q = Quat()
            q.setFromAxisAngleRad(np.pi * 0.5, Vec3(1.0, 0.0, 0.0))

            self.rest_rots.append(q)
        
        self.floorModel = loader.loadModel("models/floor/floor.bam")
        self.floorModel.reparentTo(render)
        
    def setupLights(self):
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((0.2, 0.2, 0.2, 1.0))
        
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(Vec3(0.0, 0.2, -1.0).normalized())
        directionalLight.setColor((0.9, 0.9, 0.9, 1.0))

        directionalLight.setShadowCaster(True, 1024, 1024)
        directionalLight.getLens().setNearFar(-1000.0, 1000)
        directionalLight.getLens().setFilmSize(200, 200)
        directionalLight.getLens().setViewVector(directionalLight.getDirection(), Vec3(1, 0, 0))

        self.directionalLightNode = render.attachNewNode(directionalLight)

        render.setLight(self.directionalLightNode)
        render.setLight(render.attachNewNode(ambientLight))

    def mainLoop(self, task):
        camera.setPos(Vec3(-150.0, 150.0, 100.0))
        camera.lookAt(Vec3(0.0, 0.0, 10.0))

        ANGLE_RES = self.h.getInputSize(0).z

        self.h.step(self.cs, [ self.h.getPredictionCs(0) ], False, 1.0)

        for i in range(12):
            target_angle = (self.h.getPredictionCs(0)[i] / float(ANGLE_RES - 1) * 2.0 - 1.0) * (0.3 * np.pi)
            
            self.angles[i] += 0.3 * (target_angle - self.angles[i])

        # Adjust angles
        for i in range(4):
            q1 = Quat()
            q1.setFromAxisAngleRad(self.angles[0 * 4 + i], Vec3(0.0, 1.0, 0.0))
            
            q2 = Quat()
            q2.setFromAxisAngleRad(-self.angles[1 * 4 + i], Vec3(1.0, 0.0, 0.0))

            self.limbs[i * 2 + 0].setQuat(self.rest_rots[i * 2 + 0] * q1 * q2)

            q1 = Quat()
            q1.setFromAxisAngleRad(-self.angles[2 * 4 + i], Vec3(1.0, 0.0, 0.0))

            self.limbs[i * 2 + 1].setQuat(self.rest_rots[i * 2 + 1] * q1)
        
        return task.cont

disp = PupperDisplay()
disp.run()
