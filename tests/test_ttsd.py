"""Tests that need no model: text splitting, settings validation, post-processing and the HTTP API.

Run from the repository root:  python -m unittest discover tests
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

TMP = tempfile.mkdtemp(prefix="ttsd-test-")
with open(os.path.join(TMP, "config.json"), "w") as f:
    json.dump({"port": 0, "data_dir": os.path.join(TMP, "data"), "models_dir": os.path.join(TMP, "models")}, f)
os.environ["TTSD_CONFIG"] = os.path.join(TMP, "config.json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ttsd  # noqa: E402


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TextAndSettings(unittest.TestCase):
    def test_split_on_blank_lines(self):
        self.assertEqual(ttsd.split_paragraphs("One.\n\nTwo.\n  \nThree."), ["One.", "Two.", "Three."])

    def test_split_on_lines_when_no_blank_lines(self):
        self.assertEqual(ttsd.split_paragraphs("一\n二\r\n三"), ["一", "二", "三"])

    def test_clean_settings_rejects_and_clamps(self):
        s = ttsd.clean_settings({
            "voice": "system:Nobody", "lang": "klingon", "temperature": 9, "speed": "0.1",
            "gate": None, "keep": "x", "unknown": 1, "instruct": "a" * 500,
        }, ttsd.DEFAULT_SETTINGS)
        self.assertEqual(s["voice"], ttsd.DEFAULT_SETTINGS["voice"])
        self.assertEqual(s["lang"], ttsd.DEFAULT_SETTINGS["lang"])
        self.assertEqual(s["temperature"], 1.5)
        self.assertEqual(s["speed"], 0.5)
        self.assertIsNone(s["gate"])
        self.assertEqual(s["keep"], ttsd.DEFAULT_SETTINGS["keep"])
        self.assertNotIn("unknown", s)
        self.assertEqual(len(s["instruct"]), 200)

    def test_valid_voice(self):
        self.assertTrue(ttsd.valid_voice("system:Vivian"))
        self.assertFalse(ttsd.valid_voice("system:vivian"))
        self.assertFalse(ttsd.valid_voice("custom:../../etc"))
        self.assertFalse(ttsd.valid_voice(None))

    def test_post_params(self):
        self.assertEqual(ttsd.post_params({"gate": "off"}, ttsd.DEFAULT_SETTINGS), (1.0, None, 0.05))
        self.assertEqual(ttsd.post_params({"speed": "1.2", "gate": "-35", "keep": "0.2"}, {}), (1.2, -35.0, 0.2))


@unittest.skipUnless(shutil.which("ffmpeg") or os.path.exists(ttsd.CFG["ffmpeg"]), "ffmpeg not installed")
class Render(unittest.TestCase):
    def test_breath_removal_shortens_pauses_and_keeps_lead_in(self):
        pid, tid = "a" * 10, "b" * 10
        takes = os.path.join(ttsd.pdir(pid), "takes")
        os.makedirs(takes, exist_ok=True)
        t = np.arange(int(0.5 * ttsd.SR)) / ttsd.SR
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        gap = np.zeros(int(1.0 * ttsd.SR))
        sf.write(os.path.join(takes, f"{tid}.wav"), np.concatenate([gap, tone, gap, tone]), ttsd.SR)
        out = ttsd.render_take(pid, tid, 1.0, -40, 0.05)
        a, _ = sf.read(out)
        self.assertLess(len(a) / ttsd.SR, 1.4)                  # 3.0 s → about 1.1 s
        lead = np.argmax(np.abs(a) > 0.01) / ttsd.SR
        self.assertTrue(0.02 < lead < 0.15, lead)                # a short lead-in is kept
        self.assertEqual(ttsd.render_take(pid, tid, 1.0, None, 0.05), os.path.join(takes, f"{tid}.wav"))


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for d in ("projects", "voices", "uploads"):
            os.makedirs(os.path.join(ttsd.DATA, d), exist_ok=True)
        ttsd.ENGINE = ttsd.Engine()
        cls.srv = ttsd.ThreadingHTTPServer(("127.0.0.1", 0), ttsd.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_config_and_voices(self):
        code, cfg = self.call("GET", "/api/config")
        self.assertEqual(code, 200)
        self.assertFalse(cfg["features"]["transcribe"])
        code, voices = self.call("GET", "/api/voices")
        self.assertEqual(len([v for v in voices if v["kind"] == "system"]), 9)

    def test_project_lifecycle(self):
        code, p = self.call("POST", "/api/projects", {"text": "First.\n\nSecond.", "settings": {"voice": "system:Ryan"}})
        self.assertEqual(code, 200)
        self.assertEqual([x["text"] for x in p["paragraphs"]], ["First.", "Second."])
        self.assertEqual(p["settings"]["voice"], "system:Ryan")
        first = p["paragraphs"][0]["id"]
        code, p = self.call("PUT", f"/api/projects/{p['id']}", {
            "title": "Renamed", "settings": {"gate": None},
            "paragraphs": [{"id": first, "text": "First, edited."}, {"text": "New."}]})
        self.assertEqual(p["title"], "Renamed")
        self.assertIsNone(p["settings"]["gate"])
        self.assertEqual([x["text"] for x in p["paragraphs"]], ["First, edited.", "New."])
        self.assertEqual(p["paragraphs"][0]["id"], first)
        code, err = self.call("POST", f"/api/projects/{p['id']}/export", {"format": "wav"})
        self.assertEqual((code, err["error"]), (400, "nothing_to_export"))
        self.assertEqual(self.call("DELETE", f"/api/projects/{p['id']}")[0], 200)
        self.assertEqual(self.call("GET", f"/api/projects/{p['id']}")[1]["error"], "script_missing")

    def test_errors_are_codes(self):
        self.assertEqual(self.call("GET", "/api/nope")[1]["error"], "not_found")
        self.assertEqual(self.call("POST", "/api/voices", {"name": "x"})[1]["error"], "need_ref_text")
        self.assertEqual(self.call("POST", "/api/transcribe", {"upload_id": "abcdef12"})[1]["error"], "upload_missing")

    def test_static_files_stay_inside_web_dir(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"<html", r.read())
        code, _ = self.call("GET", "/..%2F..%2Fttsd.py")
        self.assertEqual(code, 404)


if __name__ == "__main__":
    unittest.main()
