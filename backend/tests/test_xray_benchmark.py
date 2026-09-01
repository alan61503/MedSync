from pathlib import Path

from backend.evaluation.xray_benchmark import discover_benchmark_dataset


def test_discover_benchmark_dataset_binary_layout(tmp_path):
    normal_dir = tmp_path / "NORMAL"
    osteo_dir = tmp_path / "OSTEOPOROSIS"
    normal_dir.mkdir()
    osteo_dir.mkdir()
    (normal_dir / "n1.jpg").write_bytes(b"fake")
    (normal_dir / "n2.jpg").write_bytes(b"fake")
    (osteo_dir / "o1.jpg").write_bytes(b"fake")
    (osteo_dir / "o2.jpg").write_bytes(b"fake")

    dataset = discover_benchmark_dataset(str(tmp_path))

    assert dataset is not None
    assert dataset["total_images"] == 4
    assert dataset["normal_images_count"] == 2
    assert dataset["osteoporosis_images_count"] == 2
    assert len(dataset["normal_images"]) == 2
    assert len(dataset["osteoporosis_images"]) == 2
    assert dataset["label_map"] == {"NORMAL": 0, "OSTEOPOROSIS": 1}
