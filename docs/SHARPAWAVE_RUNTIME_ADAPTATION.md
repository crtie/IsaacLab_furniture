# Sharpawave runtime API and limits

The supported default is the asset-derived `floating` variant. The runtime
maps all 28 joints by exact runtime name, applies the asset drive parameters,
supports reset/step/close, and exposes joint, wrist, finger and five contact
channel observations.

The runtime canary verifies a non-zero joint command and readback without Wuji
modules. The contact-channel check places a known sphere against each finger;
the contacted channel becomes non-zero, the other channels remain zero, and all
channels recover after separation. These checks establish simulation wiring,
not hardware calibration or successful grasping.

The formal policy path additionally requires:

- a versioned calibration file;
- Pick and Insert checkpoints;
- observation/action schema and normalization metadata;
- control frequency and physical success tolerances.

If a checkpoint, manifest field, schema, normalization or calibration is absent
or invalid, the runner returns `POLICY_UNAVAILABLE`. It never falls back to
system-validation, Wuji, sticky, Oracle, snap, teleport or root-pose writes.

The external asset origin, revision and redistribution license must be supplied
by the asset owner before redistribution. Physical grasp, lift, insertion and
release success remain unverified until the corresponding calibration,
checkpoints and tolerances are supplied.
