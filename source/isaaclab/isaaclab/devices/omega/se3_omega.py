# SPDX-License-Identifier: BSD-3-Clause
# Omega.6 SE(3) delta controller using forcedimension_core.dhd

import threading, time
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from collections.abc import Callable

from forcedimension_core import dhd
from ..device_base import DeviceBase, DeviceCfg


@dataclass
class Se3Omega6Cfg(DeviceCfg):
    pos_sensitivity: float = 0.4  # scale on translation deltas [m]
    rot_sensitivity: float = 0.8  # scale on rotation deltas [rad]
    deadband_pos: float = 2e-7  # [m]
    deadband_rot: float = 2e-3  # [rad]
    sample_hz: int = 333  # best-effort sampler


class Se3Omega6(DeviceBase):
    """Force Dimension Omega.6 → SE3 delta pose + binary gripper toggle."""

    def __init__(self, cfg: Se3Omega6Cfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.deadband_pos = cfg.deadband_pos
        self.deadband_rot = cfg.deadband_rot
        self.sample_hz = max(50, int(cfg.sample_hz))
        self._sim_device = cfg.sim_device

        # open first available device; becomes default device
        self._id = dhd.open()
        if self._id < 0:
            raise OSError("Omega.6 not found. Is the device connected and calibrated?")

        # state
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)  # rotation vector
        self._close_gripper = False
        self._last_pos, self._last_R = self._read_pose()  # baseline
        self._curr_pos, self._curr_R = self._last_pos, self._last_R  # current pose
        self._additional_callbacks: dict[str, Callable] = {}

        # edge-detect for buttons
        self._b0_prev = 0
        self._b1_prev = 0

        # start sampler
        self._run = True
        self._th = threading.Thread(target=self._loop, name="Omega6Reader", daemon=True)
        self._th.start()

    def __del__(self):
        try:
            self._run = False
            if hasattr(self, "_th") and self._th.is_alive():
                self._th.join(timeout=0.5)
        finally:
            try:
                dhd.close(self._id)
            except Exception:
                pass

    def __str__(self) -> str:
        sn = None
        try:
            sn = dhd.getSerialNumber(self._id)
        except Exception:
            pass
        msg = "Omega.6 Controller for SE(3): Se3Omega6\n"
        msg += f"\tDevice ID: {self._id}" + (
            f" | S/N: {sn}\n" if (sn is not None and sn >= 0) else "\n"
        )
        msg += "\t----------------------------------------------\n"
        msg += "\tButton 0: toggle gripper (open/close)\n"
        msg += "\tButton 1: reset (re-zero deltas)\n"
        return msg

    # ---- Public API ---------------------------------------------------------
    def reset(self):
        self._close_gripper = False
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
        p, R = self._read_pose()
        if p is not None:
            self._last_pos, self._last_R = p, R
            self._curr_pos, self._curr_R = p, R

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func  # keys: "B0", "B1"

    def advance(self) -> torch.Tensor:
        # Calculate deltas here
        dpos = self._curr_pos - self._last_pos
        drot = (self._curr_R * self._last_R.inv()).as_rotvec()

        dpos = self._deadband(dpos, self.deadband_pos)
        drot = self._deadband(drot, self.deadband_rot)

        delta_pos = self.pos_sensitivity * dpos
        delta_rot = self.rot_sensitivity * drot

        g = -1.0 if self._close_gripper else 1.0
        out = np.append(np.concatenate([delta_pos, delta_rot]), g)

        # Update baseline for next advance
        self._last_pos, self._last_R = self._curr_pos, self._curr_R

        return torch.tensor(out, dtype=torch.float32, device=self._sim_device)

    # ---- Internals ----------------------------------------------------------
    def _read_pose(self):
        """Read (pos[m], Rotation) using orientation frame for Omega.6."""
        p = [0.0, 0.0, 0.0]
        Rm = [[0.0, 0.0, 0.0] for _ in range(3)]
        rc = dhd.getPositionAndOrientationFrame(p, Rm, self._id)
        if rc < 0:
            return None, None
        pos = np.array(p, dtype=float)
        R = Rotation.from_matrix(np.array(Rm, dtype=float))
        return pos, R

    def _read_buttons(self):
        b0 = dhd.getButton(0, self._id)
        b1 = dhd.getButton(1, self._id)
        return int(max(0, b0)), int(max(0, b1))

    @staticmethod
    def _deadband(v: np.ndarray, thresh: float) -> np.ndarray:
        return np.zeros_like(v) if np.linalg.norm(v) < thresh else v

    def _loop(self):
        dt = 1.0 / float(self.sample_hz)
        if self._last_pos is None:
            self._last_pos, self._last_R = np.zeros(3), Rotation.identity()
            self._curr_pos, self._curr_R = self._last_pos, self._last_R

        while self._run:
            t0 = time.time()

            # Only update current pose and rotation
            pos, R = self._read_pose()
            if pos is not None:
                self._curr_pos, self._curr_R = pos, R

            # buttons: edges
            b0, b1 = self._read_buttons()
            if b0 == 1 and self._b0_prev == 0:
                self._close_gripper = not self._close_gripper
                cb = self._additional_callbacks.get("B0")
                if cb:
                    try:
                        cb()
                    except Exception:
                        pass
            if b1 == 1 and self._b1_prev == 0:
                self.reset()
                cb = self._additional_callbacks.get("B1")
                if cb:
                    try:
                        cb()
                    except Exception:
                        pass
            self._b0_prev, self._b1_prev = b0, b1

            # rate control
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
