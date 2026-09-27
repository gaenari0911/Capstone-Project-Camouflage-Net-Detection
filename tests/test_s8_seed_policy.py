import json
import sqlite3
import unittest

from capstone_lab.campaign.contracts import readiness_plan
from capstone_lab.campaign.io import atomic_json
from capstone_lab.campaign.seed_policy import AMENDMENT, amendment, cancel_pending, excluded
from capstone_lab.campaign.supervisor import launch
from capstone_lab.errors import StateError
from tests.test_s8_campaign import ROOT, S8Integration, await_terminal, fixture_config


class SeedPolicyTests(unittest.TestCase):
    def test_plan_counts_and_references(self):
        plan = readiness_plan(ROOT)
        rows = plan["registry"]
        trains = [r for r in rows if r["node_kind"] == "TRAIN"]
        self.assertEqual(len(trains), 34)
        self.assertEqual(sum(int(r["epochs"]) for r in trains), 4440)
        self.assertEqual(sum(r["node_kind"] == "REFERENCE" for r in rows), 2)
        self.assertFalse(any(excluded(r["job_id"]) for r in rows))
        nodes = {n["node_id"]: n for n in plan["nodes"]}
        self.assertTrue(all(d in nodes for n in nodes.values() for d in n["dependencies"]))
        self.assertTrue(all(f"M0_seed{s}" in nodes and f"L0_seed{s}" in nodes for s in (0, 1, 2)))

    def test_no_partial_cancellation_when_any_job_already_started(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY, record TEXT)")
        db.execute("CREATE TABLE events(created REAL, kind TEXT, detail TEXT)")
        for name, state, attempts in (("H0_R_seed1", "PLANNED", 0), ("H1_R_seed1", "RUNNING", 1)):
            db.execute("INSERT INTO jobs VALUES(?,?)", (name, json.dumps(dict(id=name, state=state, attempts=attempts))))
        db.commit()
        policy = dict(excluded_jobs=["H0_R_seed1", "H1_R_seed1"], amendment_sha256="test")
        with self.assertRaises(StateError):
            cancel_pending(db, policy)
        self.assertEqual(json.loads(db.execute("SELECT record FROM jobs WHERE id='H0_R_seed1'").fetchone()[0])["state"], "PLANNED")
        self.assertEqual(db.execute("SELECT count(*) FROM events").fetchone()[0], 0)
        db.close()


class SeedPolicyIntegration(S8Integration):
    def test_policy_skips_only_excluded_jobs_and_finishes(self):
        config = fixture_config(self.run.relative_to(ROOT).as_posix())
        base = config["jobs"][1]
        config["jobs"] = [dict(base, id="M0_seed0", dependencies=[]),
                          dict(base, id="H0_R_seed1", dependencies=[])]
        atomic_json(self.path, config)
        _, sha = amendment(ROOT)
        atomic_json(self.run / "queue_policy.json", dict(amendment=AMENDMENT,
                    amendment_sha256=sha, excluded_jobs=["H0_R_seed1"]))
        launch(ROOT, self.path)
        result = await_terminal(self.run)
        self.assertEqual(result["status"], "SUCCEEDED")
        rows = {r["id"]: r for r in result["jobs"]}
        self.assertEqual(rows["M0_seed0"]["state"], "SUCCEEDED")
        self.assertEqual(rows["H0_R_seed1"]["state"], "CANCELLED_BY_USER")
        self.assertEqual(rows["H0_R_seed1"]["attempts"], 0)
        self.assertFalse((self.run / "jobs/H0_R_seed1").exists())
