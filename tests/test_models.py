"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean, daily_max, daily_min


ZEROES = np.array([[0, 0],
                   [0, 0],
                   [0, 0]])
INTEGERS = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])


def assert_result(test_function, test_input, test_result):
    """_summary_

    Args:
        test_function (_type_): _description_
        test_input (_type_): _description_
        test_result (_type_): _description_
    """
    npt.assert_array_equal(test_function(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected", [(ZEROES, [0, 0]), (INTEGERS, [5, 6])]
)
def test_daily_max(test, expected):
    assert_result(daily_max, test, expected)


def test_daily_min():
    assert_result(daily_min, ZEROES, [0, 0])
    assert_result(daily_min, INTEGERS, [1, 2])


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)
