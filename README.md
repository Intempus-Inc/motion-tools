# Motion tools

A desktop keyframe editor and a small library of transition curves, extracted from Intempus's robotics tools.

## Editor

Use Python 3.10 or later with Tk and a desktop OpenGL context. From a checkout:

```sh
python -m pip install '.[editor]'
intempus-motion-editor
```

The bundled SO-100 opens in a pose preview. Move the joint sliders, capture poses, set their times, and play or save the sequence. Drag the preview to orbit; scroll to zoom. `--model path/to/scene.xml` loads another fixed-base MuJoCo model with hinge and slide joints. Loading a model clears the current sequence; save it first.

The preview sets joint positions directly. It does not simulate dynamics, validate collision-free motion, or control hardware. The desktop editor was tested on macOS with Python 3.14, Tk 9, and MuJoCo 3.16.

JSON files record joint names, order, and units: radians for hinges and meters for slides. Loading requires matching joint names, order and units, with poses inside the loaded model's joint limits. Model names are informational; local file paths are not saved. Playback linearly interpolates poses, holds endpoints, and uses the last frame at a repeated timestamp. SO-100 motion-library recordings use different authoring coordinates and require an explicit conversion before use here.

## Curves

The curve library needs no third-party packages. Install with `python -m pip install .`:

```python
from intempus_motion_tools import TransitionCurveLibrary as curves

weight = curves.s_curve(0.25)
```

Inputs are finite scalars on `[0, 1]`; outputs are blend weights. See the module docstring for the Bezier and sigmoid parameters.

## License

Code: MIT. The bundled SO-100 model is derived from MuJoCo Menagerie and retains Apache 2.0; its source and modifications are recorded in [SOURCE.md](intempus_motion_tools/models/trs_so_arm100/SOURCE.md).
