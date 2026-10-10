import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workflow import TaskStore
from web_app import create_app


class TaskFoldersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "Dataset"
        self.models = self.root / "checkpoints"
        self.data.mkdir(); self.models.mkdir()
        self.patch = patch("workflow.PROJECT_ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.tmp.cleanup()

    def store(self):
        return TaskStore(self.root / "tasks.json", self.data / "missing_default")

    def folder(self, name):
        (self.data / name).mkdir()
        (self.models / name).mkdir()
        (self.data / name / "episode_000000.hdf5").write_text("test")
        (self.models / name / "policy_best.ckpt").write_text("test")

    def test_scan_stable_ids_and_remove_stale(self):
        self.folder("desktop_tidy"); self.folder("desktop_tidy_v3")
        (self.root / "tasks.json").write_text(json.dumps({"selected":"old","tasks":[{"id":"old","name":"old","dataset_dir":str(self.data/"old"),"ckpt_dir":str(self.models/"old") }]}))
        store=self.store(); rows=store.list()
        self.assertEqual({r["name"] for r in rows},{"desktop_tidy","desktop_tidy_v3"})
        ids={r["id"] for r in rows}; store.sync()
        self.assertEqual({r["id"] for r in store.list()},ids)
        self.folder("new_folder"); store.sync()
        self.assertEqual(len(store.list()),3)
        self.assertTrue(all(r["trained"] for r in store.list()))

    def test_confirm_delete_and_last_task_empty(self):
        self.folder("one"); store=self.store(); row=store.list()[0]
        plan=store.deletion_plan(row["id"])
        with self.assertRaises(ValueError): store.delete(row["id"],"wrong",plan["token"])
        self.assertTrue((self.data/"one").exists())
        result=store.delete(row["id"],"one",plan["token"])
        self.assertFalse(result["warnings"])
        self.assertFalse((self.data/"one").exists()); self.assertFalse((self.models/"one").exists())
        self.assertTrue(self.data.exists() and self.models.exists())
        self.assertIsNone(store.current(required=False)); store.sync()
        self.assertEqual(self.store().list(),[])
        created=store.create("again")
        self.assertEqual(store.current()["id"],created["id"])

    def test_shared_and_outside_directories_blocked(self):
        self.folder("a"); self.folder("b"); store=self.store()
        a,b=store.data["tasks"]
        b["ckpt_dir"]=a["ckpt_dir"]
        with self.assertRaises(ValueError): store.deletion_plan(a["id"])
        b["ckpt_dir"]=str(self.models/"b")
        a["dataset_dir"]=str(self.data)
        with self.assertRaises(ValueError): store.deletion_plan(a["id"])
        a["dataset_dir"]=str(self.root)
        with self.assertRaises(ValueError): store.deletion_plan(a["id"])

    def test_registry_write_failure_rolls_back_folders(self):
        self.folder("one"); store=self.store(); task=store.current(); plan=store.deletion_plan(task["id"])
        with patch.object(store,"_save",side_effect=OSError("disk failure")):
            with self.assertRaises(OSError): store.delete(task["id"],"one",plan["token"])
        self.assertTrue((self.data/"one/episode_000000.hdf5").exists())
        self.assertTrue((self.models/"one/policy_best.ckpt").exists())
        self.assertEqual(store.current()["id"],task["id"])

    def test_api_empty_state_and_active_guard(self):
        self.folder("one")
        app=create_app(self.data/"missing_default",task_store_path=self.root/"tasks.json")
        c=app.test_client(); row=c.get("/api/tasks").json[0]
        plan=c.get(f"/api/tasks/{row['id']}/delete-plan").json
        headers={"X-Console-Request":"1"}
        url=f"/api/tasks/{row['id']}/delete"
        body={"confirmation":"one","token":plan["token"]}
        with patch.object(app.extensions['trainer'],'active',return_value=True):
            self.assertEqual(c.post(url,json=body,headers=headers).status_code,400)
        self.assertEqual(c.post(url,json=body,headers=headers).status_code,200)
        self.assertEqual(c.get('/api/tasks').json,[])
        self.assertEqual(c.post('/api/connect',json={"mode":"demo"},headers=headers).status_code,400)
        self.assertEqual(c.post('/api/tasks',json={"name":"new"},headers=headers).status_code,201)
        app.extensions['recorder'].disconnect()
