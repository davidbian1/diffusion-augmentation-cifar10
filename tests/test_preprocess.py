import torch

from src.preprocess import create_imbalanced_dataset, SyntheticDataset


class DummyDataset:
    """Minimal (image, label) dataset stand-in, avoids downloading CIFAR-10 in tests."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.zeros(1), self.labels[idx]


def test_create_imbalanced_dataset_keeps_correct_fraction_of_target_class():
    labels = [3] * 100 + [0] * 50 + [1] * 50
    dataset = DummyDataset(labels)

    imbalanced = create_imbalanced_dataset(dataset, target_class=3, keep_fraction=0.1)

    kept_labels = [dataset.labels[i] for i in imbalanced.indices]
    assert kept_labels.count(3) == 10
    assert len(imbalanced) == 110


def test_create_imbalanced_dataset_leaves_other_classes_untouched():
    labels = [3, 3, 3, 3, 0, 1]
    dataset = DummyDataset(labels)

    imbalanced = create_imbalanced_dataset(dataset, target_class=3, keep_fraction=0.5)

    kept_labels = [dataset.labels[i] for i in imbalanced.indices]
    assert kept_labels.count(3) == 2
    assert kept_labels.count(0) == 1
    assert kept_labels.count(1) == 1


def test_create_imbalanced_dataset_keep_fraction_one_keeps_everything():
    labels = [3] * 20 + [0] * 5
    dataset = DummyDataset(labels)

    imbalanced = create_imbalanced_dataset(dataset, target_class=3, keep_fraction=1.0)

    assert len(imbalanced) == len(dataset)


def test_synthetic_dataset_applies_transform_and_returns_fixed_label():
    images = ["img_a", "img_b"]
    calls = []

    def fake_transform(img):
        calls.append(img)
        return f"transformed_{img}"

    dataset = SyntheticDataset(images, label=3, transform=fake_transform)

    assert len(dataset) == 2
    item, label = dataset[0]
    assert item == "transformed_img_a"
    assert label == 3
    assert calls == ["img_a"]
