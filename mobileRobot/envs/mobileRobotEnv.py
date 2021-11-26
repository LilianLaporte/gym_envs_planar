import numpy as np
import time
from abc import abstractmethod

from planarCommon.planarEnv import PlanarEnv


class MobileRobotEnv(PlanarEnv):


    BASE_HEIGHT = 1.0 # [m]
    BASE_WIDTH = 1.0 # [m]
    LINK_LENGTH = 1.0  # [m]

    MAX_VEL_BASE = 1
    MAX_POS_BASE = 5.0
    MAX_ACC_BASE = 10.0
    MAX_FOR_BASE = 100.0
    MAX_VEL = 4 * np.pi
    MAX_POS = np.pi
    MAX_ACC = 8 * np.pi
    MAX_TOR = 1000

    def __init__(self, render=False, n=2, dt=0.01):
        super().__init__(render=render, dt=dt)
        self._n = n
        self._limUpPos = self.joinLimits(np.array([self.MAX_POS_BASE]), np.ones(self._n-1) * self.MAX_POS)
        self._limUpVel = self.joinLimits(np.array([self.MAX_VEL_BASE]), np.ones(self._n-1) * self.MAX_VEL)
        self._limUpAcc = self.joinLimits(np.array([self.MAX_ACC_BASE]), np.ones(self._n-1) * self.MAX_ACC)
        self._limUpTor = self.joinLimits(np.array([self.MAX_FOR_BASE]), np.ones(self._n-1) * self.MAX_TOR)
        self.setSpaces()

    def joinLimits(self, limBas, limArm):
        return np.concatenate((limBas, limArm))

    @abstractmethod
    def setSpaces(self):
        pass

    def reset(self, pos=None, vel=None):
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        if self.state[0] > self.MAX_POS_BASE or self.state[0] < -self.MAX_POS_BASE:
            return True
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def forwardKinematics(self, lastLinkIndex):
        fk = np.array([self.state[0], self.BASE_HEIGHT+0.2, 0.0])
        for i in range(lastLinkIndex):
            angle = 0.0
            for j in range(i+1):
                angle += self.state[j+1]
            fk[0] += np.cos(angle) * self.LINK_LENGTH
            fk[1] += np.sin(angle) * self.LINK_LENGTH
            fk[2] += self.state[i+1]
        fk[2] += self.state[lastLinkIndex+1]
        return fk

    def render(self, mode="human"):
        bound = self.MAX_POS_BASE + 1.0
        bounds = [bound, bound]
        self.renderCommon(bounds)

        # drawAxis
        self.viewer.draw_line((-5.5, 0), (5.5, 0))

        self.renderBase()
        for i in range(self._n-1):
            self.renderLink(i)
        self.renderEndEffector()
        time.sleep(self.dt())
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def renderBase(self):
        from gym.envs.classic_control import rendering
        l, r, t, b = -0.5 * self.BASE_WIDTH, 0.5 * self.BASE_WIDTH, 0.5 * self.BASE_HEIGHT, -0.5 * self.BASE_HEIGHT
        tf = rendering.Transform(rotation=0, translation=(self.state[0], 0.5 * self.BASE_HEIGHT))
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        base = self.viewer.draw_polygon([(-0.2,-0.2), (0.0,0.0), (0.0,0.0), (0.2,-0.2)])
        baseJoint = self.viewer.draw_circle(.10)
        baseJoint.set_color(.8, .8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(self.state[0], self.BASE_HEIGHT + 0.2))
        baseJoint.add_attr(tf0)
        base.add_attr(tf0)

    def renderLink(self, i):
        from gym.envs.classic_control import rendering
        l,r,t,b = 0, self.LINK_LENGTH, .01, -.01
        fk = self.forwardKinematics(i)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
        link.set_color(0,.8, .8)
        link.add_attr(tf)
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf)

    def renderEndEffector(self):
        from gym.envs.classic_control import rendering
        fk = self.forwardKinematics(self._n-1)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self.viewer.draw_circle(.10)
        eejoint.set_color(.8, .8, 0)
        eejoint.add_attr(tf)
