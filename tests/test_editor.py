import copy
import json
import unittest

from intempus_motion_tools.editor import sample_pose, unique_object, validate_animation


class AnimationTests(unittest.TestCase):
    def setUp(self):
        self.names = ["shoulder", "slider"]
        self.units = ["rad", "m"]
        self.document = {
            "format_version": 1,
            "coordinate_system": "mujoco_qpos",
            "joint_names": self.names,
            "joint_units": self.units,
            "keyframes": [
                {"time": 1, "pose": [0, 0]},
                {"time": 2, "pose": [1, 0.2]},
                {"time": 2, "pose": [0.5, 0.1]},
                {"time": 3, "pose": [0, 0]},
            ],
        }

    def validate(self, document):
        return validate_animation(document, self.names, self.units, [(-2, 2), (-1, 1)])

    def test_interpolation_and_repeated_time_jump(self):
        frames = self.validate(self.document)
        self.assertEqual(sample_pose(frames, -1), [0, 0])
        self.assertEqual(sample_pose(frames, 1.5), [0.5, 0.1])
        self.assertEqual(sample_pose(frames, 2), [0.5, 0.1])
        self.assertEqual(sample_pose(frames, 2.5), [0.25, 0.05])
        self.assertEqual(sample_pose(frames, 10), [0, 0])
        one = [{"time": 0, "pose": [0.7, 0.3]}]
        self.assertEqual(sample_pose(one, 5), [0.7, 0.3])

    def test_model_contract_rejects_ambiguous_or_mismatched_coordinates(self):
        for key, value in (("joint_names", list(reversed(self.names))),
                           ("joint_units", ["deg", "m"]),
                           ("coordinate_system", "new_radians_empirical"),
                           ("format_version", True)):
            document = copy.deepcopy(self.document)
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(document)

    def test_bad_times_poses_and_types_are_rejected(self):
        for field, values in (("time", [-1, True, None, "1", float("inf"), float("nan"), 10**1000]),
                              ("pose", [[0], [0, 0, 0], [3, 0], [0, 2], [True, 0], [0, float("nan")]])):
            for value in values:
                document = copy.deepcopy(self.document)
                document["keyframes"][0][field] = value
                with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                    self.validate(document)
        document = copy.deepcopy(self.document)
        document["keyframes"][1]["time"] = 0
        with self.assertRaises(ValueError):
            self.validate(document)
        with self.assertRaises(ValueError):
            sample_pose(self.document["keyframes"], float("nan"))

    def test_json_round_trip_and_duplicate_key_rejection(self):
        decoded = json.loads(json.dumps(self.document), object_pairs_hook=unique_object)
        frames = self.validate(decoded)
        self.assertEqual(frames[-1]["name"], "Keyframe 4")
        frames[0]["pose"][0] = 1
        self.assertEqual(self.document["keyframes"][0]["pose"][0], 0)
        with self.assertRaises(ValueError):
            json.loads('{"time": 0, "time": 1}', object_pairs_hook=unique_object)


if __name__ == "__main__":
    unittest.main()
