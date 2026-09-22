"""Robot-only URDF FK and bounded IK; no object state or simulator oracle."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


class Arm:
    def __init__(self, urdf, names, root_pose, limits, base_link=None, ee_link=None):
        tree = ET.parse(urdf).getroot()
        self.names = list(names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate robot joint names")
        if base_link is not None and ee_link is not None:
            by_child = {joint.find("child").get("link"): joint for joint in tree.findall("joint")}
            chain, link, seen = [], ee_link, set()
            while link != base_link:
                if link in seen or link not in by_child:
                    raise ValueError("No URDF chain from base to observed end effector")
                seen.add(link)
                joint = by_child[link]
                chain.append(joint)
                link = joint.find("parent").get("link")
            chain.reverse()
        else:
            chain = [tree.find(f"joint[@name='{name}']") for name in names]
        self.joints = []
        active = []
        for joint in chain:
            if joint is None or joint.attrib["type"] not in ("revolute", "fixed"):
                raise ValueError("Expected revolute or fixed arm chain joints")
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            index, axis = None, None
            if joint.attrib["type"] == "revolute":
                index = self.names.index(joint.attrib["name"])
                active.append(index)
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis = axis/np.linalg.norm(axis)
            self.joints.append((transform, axis, index))
        if sorted(active) != list(range(len(names))):
            raise ValueError("URDF chain and observed arm joint names disagree")
        self.root = np.eye(4)
        self.root[:3, :3] = Rotation.from_quat(np.asarray(root_pose[3:])[[1, 2, 3, 0]]).as_matrix()
        self.root[:3, 3] = root_pose[:3]
        self.limits = np.asarray(limits, float)

    def fk(self, q):
        t = self.root.copy()
        if len(q) != len(self.names):
            raise ValueError("Unexpected arm joint count")
        for origin, axis, index in self.joints:
            rotation = np.eye(4)
            if index is not None:
                rotation[:3, :3] = Rotation.from_rotvec(axis*q[index]).as_matrix()
            t = t @ origin @ rotation
        return t

    def solve(self, q, position, rotation):
        def error(values):
            t = self.fk(values)
            return np.r_[t[:3, 3]-position, .1*Rotation.from_matrix(rotation @ t[:3, :3].T).as_rotvec()]
        result = least_squares(error, q, bounds=(self.limits[:, 0], self.limits[:, 1]),
                               max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        e = error(result.x)
        if np.linalg.norm(e[:3]) > .002 or np.linalg.norm(e[3:]) > .002:
            raise ValueError("Recovery target has no bounded IK solution")
        return result.x


class DualMotion:
    def __init__(self, descriptions, control_dt):
        self.arms = {name: Arm(**description) for name, description in descriptions.items()}
        self.dt = control_dt

    def validate(self, obs):
        errors = {}
        for i, name in enumerate(("left", "right")):
            pose = self.arms[name].fk(obs.state[i*7:i*7+6])
            position = np.linalg.norm(pose[:3, 3]-obs.eef_positions[i])
            measured = Rotation.from_quat(obs.eef_quaternions[i][[1, 2, 3, 0]]).as_matrix()
            rotation = np.linalg.norm(Rotation.from_matrix(pose[:3, :3] @ measured.T).as_rotvec())
            if position > .002 or rotation > .01:
                raise ValueError(f"{name} URDF FK disagrees with simulator robot observations")
            errors[name] = dict(position_m=float(position), rotation_rad=float(rotation))
        return errors

    def __call__(self, obs, plan):
        self.validate(obs)
        count = plan["duration_steps"]
        actions = np.repeat(obs.state[None], count, axis=0)
        for i, name in enumerate(("left", "right")):
            arm = self.arms[name]
            previous = obs.state[i*7:i*7+6].copy()
            start = arm.fk(previous)
            delta = np.asarray(plan[name]["translation_m"])
            rotvec = np.asarray(plan[name]["rotation_vector_rad"])
            for step in range(count):
                fraction = (step+1)/count
                rotation = Rotation.from_rotvec(rotvec*fraction).as_matrix() @ start[:3, :3]
                q = arm.solve(previous, start[:3, 3]+delta*fraction, rotation)
                if np.max(np.abs(q-previous))/self.dt > 1.5:
                    raise ValueError("Recovery exceeds 1.5 rad/s joint command slew bound")
                actions[step, i*7:i*7+6] = q
                actions[step, i*7+6] = obs.state[i*7+6] + fraction*(plan[name]["gripper_opening"]-obs.state[i*7+6])
                previous = q
        return actions
