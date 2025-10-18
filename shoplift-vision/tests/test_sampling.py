from shoplift_vision.utils.sampling import (
    strided_sample_indices,
    temporal_sampling,
    uniform_sample_indices,
)


def test_uniform_sampling_handles_short_videos() -> None:
    indices = uniform_sample_indices(total_frames=3, num_samples=5)
    assert len(indices) == 5
    assert indices[:3] == [0, 1, 2]


def test_strided_sampling_requires_stride() -> None:
    indices = strided_sample_indices(total_frames=10, num_samples=4, stride=3)
    assert indices == [0, 3, 6, 9]


def test_temporal_sampling_invalid_strategy() -> None:
    try:
        temporal_sampling(10, 4, strategy="unknown")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")
