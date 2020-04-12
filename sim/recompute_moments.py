import xmltodict
import numpy as np

d = xmltodict.parse(open("pupper.urdf", mode="r").read())

for l in d["robot"]["link"]:
    if "collision" in l:
        i = l["inertial"]["inertia"]

        m = float(l["inertial"]["mass"]["@value"])

        ixx = float(i["@ixx"])
        ixy = float(i["@ixy"])
        ixz = float(i["@ixz"])
        iyy = float(i["@iyy"])
        iyz = float(i["@iyz"])
        izz = float(i["@izz"])

        g = l["collision"]["geometry"]

        if "box" in g: # Box rotated about center
            size = np.array([ float(v) for v in g["box"]["@size"].split() ])

            i["@ixx"] = 1.0 / 12.0 * m * (size[1] ** 2 + size[2] ** 2)
            i["@ixy"] = 0.0
            i["@ixz"] = 0.0
            i["@iyy"] = 1.0 / 12.0 * m * (size[0] ** 2 + size[2] ** 2)
            i["@iyz"] = 0.0
            i["@izz"] = 1.0 / 12.0 * m * (size[0] ** 2 + size[1] ** 2)

        elif "capsule" in g: # Capsule rotated about center
            length = float(g["capsule"]["@length"])
            radius = float(g["capsule"]["@radius"])

            i["@ixx"] = 1.0 / 12.0 * m * (length ** 2)
            i["@ixy"] = 0.0
            i["@ixz"] = 0.0
            i["@iyy"] = 0.0
            i["@iyz"] = 0.0
            i["@izz"] = 1.0 / 12.0 * m * (length ** 2)

s = xmltodict.unparse(d, output=open("pupper_moment.urdf", mode="w"), pretty=True)

print("Done.")

