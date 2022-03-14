import numpy as np
import time

from forwardKinematics.planarFks.groundRobotFk import GroundRobotFk

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotArmEnv(GroundRobotEnv):
    LINK_LENGTH = 1.0
    MAX_ARM_VEL = 4 * np.pi
    MAX_ARM_POS = np.pi
    MAX_ARM_ACC = 9 * np.pi

    def __init__(self, render=False, dt=0.01, n_arm=2):
        self._n_arm = n_arm
        self._limUpArmPos = np.ones(self._n_arm) * self.MAX_ARM_POS
        self._limUpArmVel = np.ones(self._n_arm) * self.MAX_ARM_VEL
        self._limUpArmAcc = np.ones(self._n_arm) * self.MAX_ARM_ACC
        super().__init__(render=render, dt=dt)
        self._fk = GroundRobotFk(self._n_arm)

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        """ The velocity is the forward velocity and turning velocity here """
        if not isinstance(pos, np.ndarray) or not pos.size == 3 + self._n_arm:
            pos = np.zeros(3 + self._n_arm)
        if not isinstance(vel, np.ndarray) or not vel.size == 2 + self._n_arm:
            vel = np.zeros(2 + self._n_arm)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def render(self, mode="human"):
        super().render(mode=mode, final=False)
        for i in range(3, self._n_arm+3):
            self.renderLink(i)
        self.renderEndEffector()
        time.sleep(self.dt())
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def renderLink(self, i):
        from gym.envs.classic_control import rendering
        l, r, t, b = 0, self.LINK_LENGTH, 0.05, -0.05
        fk = self._fk.fk(self.state[0: self._n_arm+3], i)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, .6, .8)
        joint = self.viewer.draw_circle(.10)
        joint.set_color(0, 0.2, 0.8)
        link.add_attr(tf)
        joint.add_attr(tf)

    def renderEndEffector(self):
        from gym.envs.classic_control import rendering
        fk = self._fk.fk(self.state[0: self._n_arm+3], self._n_arm+3)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self.viewer.draw_circle(.10)
        eejoint.set_color(0, 0.2, 0.8)
        eejoint.add_attr(tf)