from src.preprocess import ExperimentConfig, parse_args


def test_parse_args_defaults_match_experiment_config_defaults():
    assert parse_args([]) == ExperimentConfig()


def test_parse_args_defaults_preserve_original_hardcoded_values():
    config = parse_args([])

    assert config.target_class == 3
    assert config.first_fraction == 0.1
    assert config.target_fractions == [0.25, 0.5, 0.75, 1.0, 2.0]
    assert config.test_batch_size == 4
    assert config.train_batch_size == 64
    assert config.repo_id == "Ketansomewhere/cifar10_conditional_diffusion1"
    assert config.save_dir == "generated_images"
    assert config.num_epochs == 5
    assert config.num_runs == 3
    assert config.lr == 1e-4
    assert config.seed == 42


def test_parse_args_overrides_are_applied():
    config = parse_args([
        "--target-class", "5",
        "--first-fraction", "0.2",
        "--target-fractions", "0.5", "1.0",
        "--num-epochs", "2",
        "--num-runs", "1",
        "--lr", "0.001",
        "--seed", "7",
    ])

    assert config.target_class == 5
    assert config.first_fraction == 0.2
    assert config.target_fractions == [0.5, 1.0]
    assert config.num_epochs == 2
    assert config.num_runs == 1
    assert config.lr == 0.001
    assert config.seed == 7
