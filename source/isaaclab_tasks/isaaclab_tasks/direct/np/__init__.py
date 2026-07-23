# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy NP environment registration boundary used by chair stage configs."""

import importlib

from .standard_registration import STANDARD_ENV_SPECS, register_standard_environments


register_standard_environments()

_COMPAT_EXPORTS = {
    "FrankaChair1Env": (".chair1_env", "FrankaChair1Env"),
    "FrankaChair2Env": (".franka_chair2_env", "FrankaChair2Env"),
    "FrankaChairTeacherEnv": (".franka_teacher_compat_env", "FrankaChairTeacherEnv"),
    "FrankaChair3Env": (".chair3_env", "FrankaChair3Env"),
    "FrankaChair4Env": (".chair4_env", "FrankaChair4Env"),
    "FrankaChair5Env": (".chair5_env", "FrankaChair5Env"),
    "FrankaChair6Env": (".chair6_env", "FrankaChair6Env"),
    "FrankaVasskar1Env": (".vasskar1_env", "FrankaVasskar1Env"),
    "FrankaVasskar2Env": (".vasskar2_env", "FrankaVasskar2Env"),
    "FrankaPlane1Env": (".plane1_env", "FrankaPlane1Env"),
    "FrankaPlane2Env": (".plane2_env", "FrankaPlane2Env"),
    "FrankaPlane3Env": (".plane3_env", "FrankaPlane3Env"),
    "FrankaPlane4Env": (".plane4_env", "FrankaPlane4Env"),
    "FrankaLego1Env": (".lego1_env", "FrankaLego1Env"),
    "FrankaLego2Env": (".lego2_env", "FrankaLego2Env"),
    "FrankaLego3Env": (".lego3_env", "FrankaLego3Env"),
    "FrankaLego4Env": (".lego4_env", "FrankaLego4Env"),
    "FrankaLego5Env": (".lego5_env", "FrankaLego5Env"),
    "FrankaLego6Env": (".lego6_env", "FrankaLego6Env"),
    "FrankaLego7Env": (".lego7_env", "FrankaLego7Env"),
}
_COMPAT_EXPORTS.update(
    {
        "SHARPAWAVE_ENV_SPECS": (".sharpawave_registration", "SHARPAWAVE_ENV_SPECS"),
        "register_sharpawave_environments": (
            ".sharpawave_registration",
            "register_sharpawave_environments",
        ),
    }
)
_COMPAT_EXPORTS.update(
    {
        name: (".np_env_cfg", name)
        for name in (
            "FrankaChair1Cfg",
            "FrankaChair2Cfg",
            "FrankaChairTeacherCfg",
            "FrankaChair3Cfg",
            "FrankaChair4Cfg",
            "FrankaChair5Cfg",
            "FrankaChair6Cfg",
            "FrankaVasskar1Cfg",
            "FrankaVasskar2Cfg",
            "FrankaPlane1Cfg",
            "FrankaPlane2Cfg",
            "FrankaPlane3Cfg",
            "FrankaPlane4Cfg",
            "FrankaLego1Cfg",
            "FrankaLego2Cfg",
            "FrankaLego3Cfg",
            "FrankaLego4Cfg",
            "FrankaLego5Cfg",
            "FrankaLego6Cfg",
            "FrankaLego7Cfg",
        )
    }
)


def __getattr__(name: str):
    """Preserve historical direct.np class imports without eager loading."""

    target = _COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = ["STANDARD_ENV_SPECS", "register_standard_environments", *_COMPAT_EXPORTS]
