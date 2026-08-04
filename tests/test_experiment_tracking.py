import json

from src.preprocess import ExperimentConfig, save_run_artifacts


def test_save_run_artifacts_writes_config_and_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(output_dir="results", num_epochs=1, num_runs=1)
    results = {-1: (0.86, 0.01), 0.0: (0.52, 0.02), 1.0: (0.60, 0.03)}

    run_dir = save_run_artifacts(config, results)

    assert run_dir.exists()
    assert run_dir.parent.name == "results"

    with open(run_dir / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["num_epochs"] == 1
    assert saved_config["num_runs"] == 1

    with open(run_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["-1"] == {"mean": 0.86, "std": 0.01}
    assert saved_metrics["1.0"] == {"mean": 0.60, "std": 0.03}

    assert (run_dir / "results.png").exists()
    assert (tmp_path / "results.png").exists()


def test_save_run_artifacts_creates_distinct_directories_per_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(output_dir="results")
    results = {-1: (0.86, 0.01), 0.0: (0.52, 0.02)}

    first_run_dir = save_run_artifacts(config, results)
    second_run_dir = save_run_artifacts(config, results)

    assert first_run_dir != second_run_dir
    assert first_run_dir.exists()
    assert second_run_dir.exists()
