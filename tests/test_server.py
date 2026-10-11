import numpy as np
from fastapi.testclient import TestClient

from whisper_diarize import server
from whisper_diarize.jobs import JobQueue
from whisper_diarize.models import ModelPool
from whisper_diarize.options import Options


def test_users_take_turns_and_positions_follow_the_schedule():
    queue = JobQueue(ModelPool(), start=False)
    audio = np.zeros(16000, np.float32)
    a1, a2, a3 = (queue.create("alice", Options()) for _ in range(3))
    b1, b2 = (queue.create("bob", Options()) for _ in range(2))
    for job in (a1, a2, a3, b1, b2):
        queue.submit(job, audio)
    queue._last_served["alice"] = 1.0  # alice went last, so bob is first in line
    assert queue.schedule() == [b1, a1, b2, a2, a3]
    assert queue.position(a3) == 5


def test_proxy_secret_is_required_when_configured(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setattr(server.settings, "proxy_secret", "s3cret")
    assert client.get("/api/config").status_code == 403
    assert client.get("/api/config", headers={"X-Proxy-Secret": "nope"}).status_code == 403
    assert client.get("/api/config", headers={"X-Proxy-Secret": "s3cret"}).status_code == 200
    monkeypatch.setattr(server.settings, "proxy_secret", None)
    assert client.get("/api/config").status_code == 200


def test_jobs_are_private_to_their_user():
    client = TestClient(server.app)
    job_id = client.post("/api/jobs", json={}, headers={"X-Remote-User": "alice"}).json()["id"]
    assert client.get(f"/api/jobs/{job_id}", headers={"X-Remote-User": "alice"}).status_code == 200
    assert client.get(f"/api/jobs/{job_id}", headers={"X-Remote-User": "bob"}).status_code == 404
    assert client.delete(f"/api/jobs/{job_id}", headers={"X-Remote-User": "bob"}).status_code == 404


def test_status_counts_work_in_progress_but_not_rejected_requests(monkeypatch):
    client = TestClient(server.app)
    state = client.get("/api/status").json()
    assert state["busy"] is False and state["recording"] == 0

    monkeypatch.setattr(server.settings, "proxy_secret", "s3cret")
    before = server.activity.last_request
    assert client.get("/api/config").status_code == 403
    assert server.activity.last_request == before

    monkeypatch.setattr(server.activity, "uploads", 1)
    state = client.get("/api/status", headers={"X-Proxy-Secret": "s3cret"}).json()
    assert state["busy"] is True and state["uploading"] == 1


def test_job_counts():
    queue = JobQueue(ModelPool(), start=False)
    queue.submit(queue.create("alice", Options()), np.zeros(16000, np.float32))
    queue.create("bob", Options())  # created, never uploaded: not work in progress
    assert queue.counts() == {"queued": 1, "running": 0, "unclaimed": 0}
