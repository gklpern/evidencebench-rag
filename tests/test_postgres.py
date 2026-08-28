import math

import pytest

from evidencebench.postgres import vector_literal


def test_vector_literal_is_stable() -> None:
    assert vector_literal((0.1, -2.0)) == "[0.1,-2]"


def test_vector_literal_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        vector_literal((math.nan,))
