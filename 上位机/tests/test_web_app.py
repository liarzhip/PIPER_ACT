import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from web_app import create_app


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(self.tmp.name)
        self.client = self.app.test_client()
        self.recorder = self.app.extensions["recorder"]

    def tearDown(self):
        self.recorder.disconnect()
        self.tmp.cleanup()

    def post(self, url, data=None):
        return self.client.post(url, json=data or {}, headers={"X-Console-Request": "1"})

    def wait_for(self, condition, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(.03)
        self.fail(f"timed out: {self.recorder.snapshot()}")

    def connect(self, **kwargs):
        self.assertEqual(self.post("/api/connect", dict(mode="demo", **kwargs)).status_code, 202)
        self.wait_for(lambda: self.recorder.snapshot()["state"] == "ready")

    def record(self):
        self.assertEqual(self.post("/api/start").status_code, 200)
        self.wait_for(lambda: self.recorder.snapshot()["count"] >= 5)
        self.assertEqual(self.post("/api/stop").status_code, 200)

    def test_end_to_end_record_play_validate_export(self):
        self.connect()
        self.assertEqual(self.client.get("/api/preview.jpg").mimetype, "image/jpeg")
        self.record()
        rows = self.client.get("/api/episodes?mode=demo").json
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["sim"])
        self.assertEqual(self.client.get("/api/episodes").json, [])
        name = rows[0]["name"]
        base = "/api/episodes/" + name
        d = self.client.get(base + "/data?mode=demo").json
        self.assertEqual(len(d["qpos"][0]), 7)
        image = self.client.get(base + "/frame/0?mode=demo")
        self.assertEqual(image.mimetype, "image/jpeg")
        self.assertTrue(image.data.startswith(b"\xff\xd8"))
        self.assertEqual(self.client.get(base + "/frame/999999?mode=demo").status_code, 400)
        report = self.post(base + "/validate?mode=demo", {"plots": True})
        self.assertEqual(report.status_code, 200, report.json)
        self.assertIn(report.json["status"], ("pass", "review"), report.json["report"])
        self.assertIn("RESULT:", report.json["report"])
        self.assertEqual(len(report.json["plots"]), 7)
        png = self.client.get(base + "/plots/j1_qpos_action.png?mode=demo")
        self.assertEqual(png.mimetype, "image/png")
        png.close()
        download = self.client.get(base + "/download?mode=demo")
        self.assertIn("attachment", download.headers["Content-Disposition"])
        download.close()
        with h5py.File(Path(self.tmp.name) / "demo" / name) as f:
            self.assertEqual(f["action"].shape[0], f.attrs["num_samples"])
            self.assertTrue(np.all(np.diff(f["timestamps/sample_monotonic"][:]) > 0))
            self.assertEqual(f["observations/images/wrist"].dtype, np.dtype("uint8"))
            if "depth" in f["observations"]:
                self.assertEqual(f["observations/depth/wrist"].dtype, np.dtype("uint16"))
                self.assertEqual(f["observations/depth/wrist"].shape[:3], f["observations/images/wrist"].shape[:3])
        self.record()
        self.assertEqual(len(self.client.get("/api/episodes?mode=demo").json), 2)
        self.assertFalse(list(Path(self.tmp.name).rglob("*.partial")))

    def test_auto_stop_disconnect_save_and_conflicting_commands(self):
        self.assertEqual(self.post("/api/start").status_code, 400)
        self.connect(duration=.3)
        self.assertEqual(self.post("/api/connect", {"mode": "demo"}).status_code, 400)
        self.post("/api/start")
        self.assertEqual(self.post("/api/start").status_code, 400)
        self.wait_for(lambda: self.recorder.snapshot()["state"] == "ready")
        self.assertEqual(len(self.client.get("/api/episodes?mode=demo").json), 1)
        self.post("/api/start")
        self.wait_for(lambda: self.recorder.snapshot()["count"] > 0)
        self.post("/api/disconnect")
        self.assertEqual(self.recorder.snapshot()["state"], "disconnected")
        self.assertEqual(len(self.client.get("/api/episodes?mode=demo").json), 2)

    def test_invalid_files_and_request_guards(self):
        environments = self.client.get("/api/environments")
        self.assertEqual(environments.status_code, 200)
        selected = next(row for row in environments.json if row["selected"])
        switched = self.post("/api/environments/select", {"name": selected["name"]})
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(switched.json["python"], selected["python"])
        self.assertEqual(self.client.post("/api/start", json={}).status_code, 403)
        self.assertEqual(self.client.post("/api/start", json={}, headers={"X-Console-Request":"1", "Origin":"https://evil.example"}).status_code, 403)
        for opts in [{"fps": 0}, {"camera_name":"../outside"}, {"mode":"invalid"}, {"fps":float("nan")}, {"wrist_height":.5}]:
            self.assertEqual(self.post("/api/connect", opts).status_code, 400)
        folder = Path(self.tmp.name)
        with h5py.File(folder / "episode_000000.hdf5", "w") as f:
            f.create_dataset("wrong", data=[1])
        report = self.post("/api/episodes/episode_000000.hdf5/validate").json
        self.assertEqual(report["status"], "fail")
        self.assertIn("Missing required dataset_stand_bottle", report["report"])
        self.assertEqual(self.client.get("/api/episodes/not_hdf5/download").status_code, 400)
        self.assertEqual(self.client.get("/api/episodes?mode=invalid").status_code, 400)

    def test_task_workflow_create_select_and_phase_guards(self):
        rows = self.client.get("/api/tasks").json
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["selected"])
        created = self.post("/api/tasks", {"name": "抓取蓝色方块"})
        self.assertEqual(created.status_code, 201, created.json)
        self.assertEqual(created.json["name"], "抓取蓝色方块")
        rows = self.client.get("/api/tasks").json
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(bool(r["selected"]) for r in rows), 1)
        selected = next(r for r in rows if r["selected"])
        self.assertRegex(selected["folder_suffix"], r"^[0-9a-f]{6}$")
        old_dataset = Path(selected["dataset_dir"])
        old_checkpoint = Path(selected["ckpt_dir"])
        (old_dataset / "dataset-marker.txt").write_text("dataset", encoding="utf-8")
        (old_checkpoint / "model-marker.txt").write_text("model", encoding="utf-8")
        renamed = self.post(f"/api/tasks/{selected['id']}/rename", {"name": "抓取绿色方块"})
        self.assertEqual(renamed.status_code, 200, renamed.json)
        self.assertEqual(renamed.json["name"], "抓取绿色方块")
        after = next(r for r in self.client.get("/api/tasks").json if r["selected"])
        new_dataset = Path(after["dataset_dir"])
        new_checkpoint = Path(after["ckpt_dir"])
        self.assertNotEqual(after["dataset_folder"], selected["dataset_folder"])
        self.assertEqual(new_dataset.name, new_checkpoint.name)
        self.assertTrue(new_dataset.name.endswith(f"_{selected['folder_suffix']}"))
        self.assertEqual((new_dataset / "dataset-marker.txt").read_text(encoding="utf-8"), "dataset")
        self.assertEqual((new_checkpoint / "model-marker.txt").read_text(encoding="utf-8"), "model")
        self.assertFalse(old_dataset.exists())
        self.assertFalse(old_checkpoint.exists())
        self.assertEqual(self.recorder.dataset, new_dataset)
        self.assertEqual(self.post("/api/training/start", {}).status_code, 400)
        self.assertEqual(self.post("/api/execution/start", {}).status_code, 400)
        self.connect()
        original = next(r for r in rows if r["name"] == "默认任务")
        self.assertEqual(self.post(f"/api/tasks/{original['id']}/select").status_code, 400)


if __name__ == "__main__":
    unittest.main()
