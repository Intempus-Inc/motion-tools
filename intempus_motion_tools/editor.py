#!/usr/bin/env python3
"""Author joint-position keyframes with a MuJoCo pose preview."""

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import mujoco
from PIL import Image, ImageTk


def finite_number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_animation(value, joint_names, joint_units, limits=None):
    """Validate the editor format against the currently loaded model."""
    if not isinstance(value, dict):
        raise ValueError("Animation must be an object")
    if type(value.get("format_version")) is not int or value["format_version"] != 1:
        raise ValueError("Expected format_version 1")
    if value.get("coordinate_system") != "mujoco_qpos":
        raise ValueError("Expected MuJoCo joint positions, not empirical motor coordinates")
    if value.get("joint_names") != joint_names or value.get("joint_units") != joint_units:
        raise ValueError("Joint names, order and units must match the loaded model")
    frames = value.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Animation must contain keyframes")
    clean = []
    previous = 0.0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Each keyframe must be an object")
        timestamp = finite_number(frame.get("time"), "Keyframe time")
        if timestamp < previous:
            raise ValueError("Keyframe times must be nonnegative and nondecreasing")
        pose = frame.get("pose")
        if not isinstance(pose, list) or len(pose) != len(joint_names):
            raise ValueError("Each pose must have one value per joint")
        pose = [finite_number(number, "Joint position") for number in pose]
        if limits is not None:
            for name, number, bound in zip(joint_names, pose, limits):
                if bound is not None and not bound[0] <= number <= bound[1]:
                    raise ValueError(f"{name} is outside the model's joint limits")
        name = frame.get("name", f"Keyframe {index + 1}")
        if not isinstance(name, str):
            raise ValueError("Keyframe name must be text")
        clean.append({"time": timestamp, "pose": pose, "name": name})
        previous = timestamp
    return clean


def sample_pose(frames, timestamp):
    """Sample validated frames; hold endpoints and use the last tied frame."""
    timestamp = finite_number(timestamp, "Sample time")
    if not frames:
        raise ValueError("Cannot sample an empty animation")
    index = bisect_right([frame["time"] for frame in frames], timestamp) - 1
    if index < 0:
        return list(frames[0]["pose"])
    if index == len(frames) - 1:
        return list(frames[-1]["pose"])
    left, right = frames[index:index + 2]
    weight = (timestamp - left["time"]) / (right["time"] - left["time"])
    return [(1 - weight) * a + weight * b for a, b in zip(left["pose"], right["pose"])]


def default_model_path():
    return Path(__file__).resolve().parent / "models/trs_so_arm100/scene.xml"


class KeyframeAnimator:
    """A desktop editor for fixed-base models with hinge and slide joints."""

    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("Intempus Motion Editor")
        self.root.geometry("1120x850")
        self.model = self.data = self.renderer = None
        self.preview_job = self.playback_job = None
        self.is_playing = False
        self.keyframes = []
        self.joint_vars = []
        self.joint_names = []
        self.joint_units = []
        self.current_time = 0.0
        self.current_robot_file = str(model_path or default_model_path())
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.load_robot()

    def setup_gui(self):
        layout = ttk.Frame(self.root, padding=16)
        layout.pack(fill="both", expand=True)
        left = ttk.Frame(layout)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))
        right = ttk.Frame(layout)
        right.pack(side="right", fill="both", expand=True)
        model_row = ttk.Frame(left)
        model_row.pack(fill="x")
        self.robot_file_var = tk.StringVar(value=self.current_robot_file)
        ttk.Entry(model_row, textvariable=self.robot_file_var).pack(side="left", fill="x", expand=True)
        ttk.Button(model_row, text="Browse", command=self.browse_robot_file).pack(side="left")
        ttk.Button(model_row, text="Load", command=self.load_robot).pack(side="left")
        self.preview = ttk.Label(left)
        self.preview.pack(pady=(12, 4))
        self.preview.bind("<ButtonPress-1>", lambda event: setattr(self, "drag_position", (event.x, event.y)))
        self.preview.bind("<B1-Motion>", self.orbit)
        self.preview.bind("<MouseWheel>", lambda event: self.zoom(0.9 if event.delta > 0 else 1.1))
        self.preview.bind("<Button-4>", lambda event: self.zoom(0.9))
        self.preview.bind("<Button-5>", lambda event: self.zoom(1.1))
        ttk.Label(left, text="Drag to orbit. Scroll to zoom. Pose preview only.").pack()
        controls = ttk.LabelFrame(left, text="Joint positions", padding=8)
        controls.pack(fill="both", expand=True, pady=12)
        canvas = tk.Canvas(controls, highlightthickness=0, width=450)
        scroll = ttk.Scrollbar(controls, orient="vertical", command=canvas.yview)
        self.joints_scroll_frame = ttk.Frame(canvas)
        self.joints_scroll_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.joints_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        pose_row = ttk.Frame(left)
        pose_row.pack(fill="x")
        ttk.Button(pose_row, text="Reset pose", command=self.reset_pose).pack(side="left")
        ttk.Button(pose_row, text="Home pose", command=self.go_to_home).pack(side="left")
        ttk.Label(right, text="Keyframes").pack(anchor="w")
        self.keyframe_listbox = tk.Listbox(right, height=15, exportselection=False)
        self.keyframe_listbox.pack(fill="both", expand=True, pady=8)
        self.keyframe_listbox.bind("<<ListboxSelect>>", self.on_keyframe_select)
        row = ttk.Frame(right)
        row.pack(fill="x")
        for label, action in (("Capture", self.capture_keyframe), ("Delete", self.delete_keyframe), ("Clear", self.clear_keyframes)):
            ttk.Button(row, text=label, command=action).pack(side="left")
        timing = ttk.Frame(right)
        timing.pack(fill="x", pady=8)
        ttk.Label(timing, text="Selected time (s)").pack(side="left")
        self.frame_time = tk.StringVar()
        ttk.Entry(timing, textvariable=self.frame_time, width=10).pack(side="left", padx=8)
        ttk.Button(timing, text="Set time", command=self.set_frame_time).pack(side="left")
        playback = ttk.Frame(right)
        playback.pack(fill="x", pady=8)
        ttk.Button(playback, text="Play", command=self.play_animation).pack(side="left")
        ttk.Button(playback, text="Stop", command=self.stop_animation).pack(side="left")
        ttk.Label(playback, text="Speed").pack(side="left", padx=(12, 0))
        self.speed_var = tk.DoubleVar(value=1)
        tk.Scale(playback, from_=0.1, to=3, resolution=0.1, variable=self.speed_var, orient="horizontal", length=130).pack(side="left")
        files = ttk.Frame(right)
        files.pack(fill="x", pady=8)
        ttk.Button(files, text="Save JSON", command=self.save_animation).pack(side="left")
        ttk.Button(files, text="Load JSON", command=self.load_animation).pack(side="left")
        self.status_var = tk.StringVar(value="Loading model")
        ttk.Label(self.root, textvariable=self.status_var, padding=(16, 8), wraplength=1050).pack(fill="x")

    def browse_robot_file(self):
        path = filedialog.askopenfilename(title="Select model", filetypes=[("MuJoCo XML", "*.xml")], initialdir=str(Path(self.current_robot_file).parent))
        if path:
            self.robot_file_var.set(path)

    def load_robot(self):
        path = self.robot_file_var.get()
        renderer = None
        try:
            model = mujoco.MjModel.from_xml_path(path)
            scalar_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
            if model.njnt == 0 or any(int(kind) not in scalar_types for kind in model.jnt_type):
                raise ValueError("Use a fixed-base model containing only hinge and slide joints")
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=300, width=480)
        except Exception as error:
            if renderer is not None:
                renderer.close()
            messagebox.showerror("Cannot load model", str(error))
            return
        self.close_preview()
        self.model, self.data, self.renderer = model, data, renderer
        self.current_robot_file = path
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}" for index in range(model.njnt)]
        self.joint_units = ["rad" if kind == mujoco.mjtJoint.mjJNT_HINGE else "m" for kind in model.jnt_type]
        self.limits = [tuple(model.jnt_range[index]) if model.jnt_limited[index] else None for index in range(model.njnt)]
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = model.stat.center
        self.camera.distance = max(0.1, 2 * model.stat.extent)
        self.camera.azimuth, self.camera.elevation = 135, -25
        for widget in self.joints_scroll_frame.winfo_children():
            widget.destroy()
        self.joint_vars = []
        self.joint_sliders = []
        for name, unit, bound in zip(self.joint_names, self.joint_units, self.limits):
            row = ttk.Frame(self.joints_scroll_frame)
            row.pack(fill="x")
            ttk.Label(row, text=f"{name} ({unit})", width=22).pack(side="left")
            var = tk.DoubleVar(value=0)
            self.joint_vars.append(var)
            lower, upper = bound or ((-math.pi, math.pi) if unit == "rad" else (-1, 1))
            slider = tk.Scale(row, from_=lower, to=upper, resolution=-1, digits=12, orient="horizontal", variable=var, length=220, command=self.manual_pose)
            slider.pack(side="left")
            self.joint_sliders.append(slider)
        self.keyframes = []
        self.update_keyframe_list()
        self.go_to_home()
        self.status_var.set(f"{Path(path).name}: {model.njnt} joints. Capture poses, then set their times.")
        self.refresh_preview()

    def orbit(self, event):
        if self.renderer is None:
            return
        x, y = self.drag_position
        self.camera.azimuth -= (event.x - x) * 0.5
        self.camera.elevation = max(-89, min(89, self.camera.elevation - (event.y - y) * 0.5))
        self.drag_position = (event.x, event.y)

    def zoom(self, factor):
        if self.renderer is not None:
            self.camera.distance = max(0.01, min(1000, self.camera.distance * factor))

    def refresh_preview(self):
        self.preview_job = None
        if self.renderer is None:
            return
        self.renderer.update_scene(self.data, camera=self.camera)
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
        self.preview.configure(image=self.preview_image)
        self.preview_job = self.root.after(33, self.refresh_preview)

    def apply_slider_positions(self):
        if self.model is None:
            return
        for index, var in enumerate(self.joint_vars):
            self.data.qpos[self.model.jnt_qposadr[index]] = var.get()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def manual_pose(self, value=None):
        if not self.is_playing:
            self.apply_slider_positions()

    def apply_pose(self, pose):
        for var, slider, value, bound in zip(self.joint_vars, self.joint_sliders, pose, self.limits):
            if bound is None:
                slider.configure(from_=min(float(slider.cget("from")), value), to=max(float(slider.cget("to")), value))
            var.set(value)
        self.apply_slider_positions()

    def reset_pose(self):
        if self.model is not None:
            self.stop_animation()
            self.apply_pose([self.model.qpos0[address] for address in self.model.jnt_qposadr])

    def go_to_home(self):
        if self.model is not None:
            self.stop_animation()
            key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            pose = self.model.key_qpos[key] if key >= 0 else self.model.qpos0
            self.apply_pose([pose[address] for address in self.model.jnt_qposadr])

    def capture_keyframe(self):
        if not self.joint_vars:
            return
        self.stop_animation()
        timestamp = self.keyframes[-1]["time"] + 1 if self.keyframes else 0
        self.keyframes.append({"time": timestamp, "pose": [var.get() for var in self.joint_vars], "name": f"Keyframe {len(self.keyframes) + 1}"})
        self.update_keyframe_list()

    def delete_keyframe(self):
        selected = self.keyframe_listbox.curselection()
        if selected:
            self.stop_animation()
            self.keyframes.pop(selected[0])
            self.update_keyframe_list()

    def clear_keyframes(self):
        if self.keyframes and messagebox.askyesno("Clear animation", "Remove all keyframes?"):
            self.stop_animation()
            self.keyframes = []
            self.update_keyframe_list()

    def set_frame_time(self):
        selected = self.keyframe_listbox.curselection()
        if not selected:
            return
        try:
            timestamp = finite_number(float(self.frame_time.get()), "Time")
            if timestamp < 0:
                raise ValueError("Time must be nonnegative")
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Invalid time", str(error))
            return
        self.stop_animation()
        self.keyframes[selected[0]]["time"] = timestamp
        self.keyframes.sort(key=lambda frame: frame["time"])
        self.update_keyframe_list()

    def update_keyframe_list(self):
        self.keyframe_listbox.delete(0, tk.END)
        for frame in self.keyframes:
            self.keyframe_listbox.insert(tk.END, f'{frame["time"]:g}s  {frame["name"]}')
        self.frame_time.set("")

    def on_keyframe_select(self, event=None):
        selected = self.keyframe_listbox.curselection()
        if selected:
            self.stop_animation()
            frame = self.keyframes[selected[0]]
            self.frame_time.set(str(frame["time"]))
            self.apply_pose(frame["pose"])

    def play_animation(self):
        if not self.keyframes:
            return
        self.stop_animation()
        self.is_playing = True
        self.current_time = 0
        self.last_tick = time.monotonic()
        self.run_animation()

    def stop_animation(self):
        self.is_playing = False
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

    def run_animation(self):
        self.playback_job = None
        if not self.is_playing:
            return
        now = time.monotonic()
        self.current_time += (now - self.last_tick) * self.speed_var.get()
        self.last_tick = now
        duration = self.keyframes[-1]["time"]
        self.apply_pose(sample_pose(self.keyframes, min(self.current_time, duration)))
        if self.current_time >= duration:
            self.stop_animation()
        else:
            self.playback_job = self.root.after(16, self.run_animation)

    def animation_document(self):
        return {"format_version": 1, "coordinate_system": "mujoco_qpos", "model": Path(self.current_robot_file).name, "joint_names": self.joint_names, "joint_units": self.joint_units, "keyframes": self.keyframes}

    def save_animation(self):
        if not self.keyframes:
            return
        path = filedialog.asksaveasfilename(title="Save animation", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                document = self.animation_document()
                validate_animation(document, self.joint_names, self.joint_units, self.limits)
                Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                self.status_var.set(f"Saved {Path(path).name}")
            except (ValueError, OSError) as error:
                messagebox.showerror("Cannot save animation", str(error))

    def load_animation(self):
        if self.model is None:
            messagebox.showerror("No model", "Load a model before loading an animation")
            return
        path = filedialog.askopenfilename(title="Load animation", filetypes=[("JSON", "*.json")])
        if path:
            try:
                document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
                frames = validate_animation(document, self.joint_names, self.joint_units, self.limits)
            except (ValueError, OSError) as error:
                messagebox.showerror("Cannot load animation", str(error))
                return
            self.stop_animation()
            self.keyframes = frames
            self.update_keyframe_list()
            self.apply_pose(frames[0]["pose"])
            self.status_var.set(f"Loaded {Path(path).name}")

    def close_preview(self):
        self.stop_animation()
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def on_closing(self):
        self.close_preview()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="MuJoCo XML model; defaults to the bundled SO-100")
    args = parser.parse_args()
    KeyframeAnimator(args.model).run()


if __name__ == "__main__":
    main()
