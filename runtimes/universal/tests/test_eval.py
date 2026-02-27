"""Tests for vision model evaluation."""

from pathlib import Path

import pytest

from models.eval_model import EvalDB, EvalResult, ModelEvaluator


class TestEvalResult:
    def test_compute_score_defaults(self):
        r = EvalResult(
            model_name="test", model_path="/tmp/test.pt", dataset="data.yaml",
            mAP50=0.5, mAP50_95=0.3, f1=0.4,
            small_object_recall=0.2, inference_ms=100,
        )
        score = r.compute_score()
        assert score > 0
        assert r.score == score

    def test_compute_score_custom_weights(self):
        r = EvalResult(
            model_name="test", model_path="/tmp/test.pt", dataset="data.yaml",
            mAP50=1.0, mAP50_95=1.0, f1=1.0,
            small_object_recall=1.0, inference_ms=10,
        )
        score = r.compute_score({"mAP50_95": 1.0, "mAP50": 0, "small_object_recall": 0, "f1": 0, "speed": 0})
        assert abs(score - 1.0) < 1e-6

    def test_compute_score_speed_penalty(self):
        fast = EvalResult(model_name="fast", model_path="", dataset="",
                          mAP50=0.5, mAP50_95=0.3, f1=0.4,
                          small_object_recall=0.2, inference_ms=20)
        slow = EvalResult(model_name="slow", model_path="", dataset="",
                          mAP50=0.5, mAP50_95=0.3, f1=0.4,
                          small_object_recall=0.2, inference_ms=600)
        fast.compute_score()
        slow.compute_score()
        assert fast.score > slow.score

    def test_to_dict(self):
        r = EvalResult(model_name="test", model_path="/tmp/test.pt", dataset="d.yaml")
        d = r.to_dict()
        assert d["model_name"] == "test"
        assert "mAP50" in d
        assert "per_class" in d


class TestEvalDB:
    @pytest.fixture
    def db(self, tmp_path):
        return EvalDB(tmp_path / "test_eval.db")

    def test_save_and_leaderboard(self, db):
        r1 = EvalResult(model_name="model_a", model_path="a.pt", dataset="d.yaml",
                         timestamp="2026-01-01", score=0.8, mAP50=0.7)
        r2 = EvalResult(model_name="model_b", model_path="b.pt", dataset="d.yaml",
                         timestamp="2026-01-02", score=0.9, mAP50=0.8)
        db.save(r1)
        db.save(r2)
        lb = db.leaderboard()
        assert len(lb) == 2
        assert lb[0]["model_name"] == "model_b"  # Higher score first

    def test_best_model(self, db):
        r = EvalResult(model_name="best", model_path="b.pt", dataset="d.yaml",
                        timestamp="2026-01-01", score=0.95)
        db.save(r)
        best = db.best_model()
        assert best is not None
        assert best["model_name"] == "best"

    def test_best_model_empty(self, db):
        assert db.best_model() is None

    def test_leaderboard_filter_dataset(self, db):
        r1 = EvalResult(model_name="a", model_path="", dataset="ds1", score=0.5)
        r2 = EvalResult(model_name="b", model_path="", dataset="ds2", score=0.6)
        db.save(r1)
        db.save(r2)
        lb = db.leaderboard(dataset="ds1")
        assert len(lb) == 1
        assert lb[0]["dataset"] == "ds1"

    def test_record_promotion(self, db):
        r = EvalResult(model_name="promo", model_path="", dataset="", score=0.9)
        eval_id = db.save(r)
        db.record_promotion("promo", eval_id, "auto-promoted")
        # Verify in DB
        row = db._db.execute("SELECT * FROM promotions WHERE model_name = ?", ("promo",)).fetchone()
        assert row is not None
        assert row["reason"] == "auto-promoted"

    def test_model_history(self, db):
        for i in range(5):
            r = EvalResult(model_name="tracked", model_path="", dataset="",
                            timestamp=f"2026-01-0{i+1}", score=0.1 * i)
            db.save(r)
        history = db.model_history("tracked")
        assert len(history) == 5

    def test_per_class_json_roundtrip(self, db):
        r = EvalResult(model_name="pc", model_path="", dataset="",
                        per_class={"person": 0.8, "car": 0.6})
        db.save(r)
        lb = db.leaderboard()
        assert lb[0]["per_class"] == {"person": 0.8, "car": 0.6}


class TestModelEvaluatorCompare:
    def test_compare_picks_winner(self, tmp_path):
        ev = ModelEvaluator(db=EvalDB(tmp_path / "test_eval_cmp.db"))
        a = EvalResult(model_name="a", model_path="", dataset="", score=0.8,
                        mAP50=0.7, mAP50_95=0.5, precision=0.6, recall=0.5,
                        f1=0.55, inference_ms=50, small_object_recall=0.3)
        b = EvalResult(model_name="b", model_path="", dataset="", score=0.6,
                        mAP50=0.5, mAP50_95=0.3, precision=0.4, recall=0.3,
                        f1=0.35, inference_ms=100, small_object_recall=0.1)
        comp = ev.compare(a, b)
        assert comp["winner"] == "a"
        assert comp["score_diff"] == pytest.approx(0.2)
        assert comp["metrics"]["mAP50"]["better"] == "a"
        assert comp["metrics"]["inference_ms"]["better"] == "a"  # lower is better

    def test_compare_speed_better_for_lower(self, tmp_path):
        ev = ModelEvaluator(db=EvalDB(tmp_path / "test_eval_cmp2.db"))
        a = EvalResult(model_name="a", model_path="", dataset="", score=0.5, inference_ms=200)
        b = EvalResult(model_name="b", model_path="", dataset="", score=0.5, inference_ms=50)
        comp = ev.compare(a, b)
        assert comp["metrics"]["inference_ms"]["better"] == "b"
