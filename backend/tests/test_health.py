"""本地自检：确认后端骨架能起来、/health 能通。

不需要启动服务器，直接用 FastAPI 的 TestClient。
运行：
    cd backend
    python -m tests.test_health
"""

from fastapi.testclient import TestClient

from app.config import APP_VERSION, FAULT_NAMES, FEATURE_NAMES_SMALL, NUM_CLASSES
from main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == APP_VERSION
    assert "time" in data
    print("[PASS] /health 返回 200 且 status=ok")


def test_root_points_to_docs() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"
    print("[PASS] / 指向 /docs")


def test_config_is_consistent() -> None:
    assert len(FAULT_NAMES) == 6, f"必须有 6 个故障类别，实际 {len(FAULT_NAMES)}"
    assert NUM_CLASSES == 6
    assert len(FEATURE_NAMES_SMALL) == 8, "8 维特征版必须有 8 列"
    assert FEATURE_NAMES_SMALL[0] == "cpu_usage_pct"
    assert FEATURE_NAMES_SMALL[-1] == "error_rate"
    print("[PASS] 配置一致：6 类标签、8 个特征")


def main() -> None:
    test_health_returns_ok()
    test_root_points_to_docs()
    test_config_is_consistent()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
