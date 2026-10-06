from fastapi.testclient import TestClient

from loop.api.main import app
from loop.schema import REPLY_CLASSES

client = TestClient(app)

def test_api_health_contract() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_api_customers_contract() -> None:
    res = client.get("/api/customers")
    assert res.status_code == 200
    data = res.json()
    assert "customers" in data
    assert isinstance(data["customers"], list)
    if len(data["customers"]) > 0:
        c = data["customers"][0]
        assert "customer_id" in c
        assert "name" in c
        assert "industry" in c
        assert "primary_language" in c
        assert "n_sends" in c

def test_api_classify_contract() -> None:
    payload = {"text": "Sounds compelling. Can you send over your calendar link?"}
    res = client.post("/api/replies/classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in REPLY_CLASSES
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == len(REPLY_CLASSES)
    assert "draft_reply" in data
    assert "validators" in data
    assert isinstance(data["validators"], dict)
    assert "routed_to_review" in data
    assert "model" in data

def test_api_evaluation_contract() -> None:
    res = client.get("/api/evaluation")
    assert res.status_code in {200, 404}
    if res.status_code == 404:
        data = res.json()
        assert "detail" in data
