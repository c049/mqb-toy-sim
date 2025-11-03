from mqb_toy.analysis.sweeps import grid_scan, random_scan


def test_grid_scan_returns_metrics():
    grid = grid_scan(
        g_values=[0.1, 0.2],
        delta_values=[0.1, 0.2],
        base_config={"N": 6, "steps": 10, "dt": 0.05},
    )
    assert len(grid) == 4
    sample = grid[0]
    assert "purity_final" in sample and "coherence_final" in sample


def test_random_scan_sample_size():
    res = random_scan(
        num_samples=3,
        g_range=(0.1, 0.3),
        delta_range=(0.05, 0.2),
        base_config={"N": 6, "steps": 8, "dt": 0.05},
        seed=0,
    )
    assert len(res) == 3
