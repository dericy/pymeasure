#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2026 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
import pytest

from pymeasure.test import expected_protocol
from pymeasure.instruments.srs.sr400 import SR400
# CountingMode


@pytest.mark.parametrize(
    "s,idx",
    [
        (SR400.CountingMode.A_AND_B, 0),
        (SR400.CountingMode.A_MINUS_B, 1),
        (SR400.CountingMode.A_PLUS_B, 2),
        (SR400.CountingMode.A_FOR_B, 3),
    ],
)
def test_counting_mode(s, idx):
    """Verify the communication of the sensitivity getter."""
    with expected_protocol(
        SR400,
        [
            (f"CM {idx}", None),
            ("CM", f"{idx}"),
        ],
    ) as inst:
        inst.counting_mode = s
        assert inst.counting_mode == s


@pytest.mark.parametrize("val", [0, 0.002, 1.5, 60])
def test_dwell_time_valid(val):
    with expected_protocol(SR400, [(f"DT {val:g}", None), ("DT", f"{val:g}")]) as inst:
        inst.dwell_time = val
        assert inst.dwell_time == val


def test_dwell_time_invalid():
    with expected_protocol(SR400, []) as inst:
        with pytest.raises(ValueError):
            inst.dwell_time = 0.001  # Too small (not 0)
        with pytest.raises(ValueError):
            inst.dwell_time = 61  # Too large


@pytest.mark.parametrize("label, idx", [("A", 0), ("B", 1), ("A-B", 2), ("A+B", 3)])
def test_dac_output(label, idx):
    # This assumes map_values=True is in your Instrument.setting
    with expected_protocol(SR400, [(f"AS {idx}", None)]) as inst:
        inst.dac_output = label


def test_dac_output_invalid():
    with expected_protocol(SR400, []) as inst:
        with pytest.raises(ValueError):
            inst.dac_output = "C"  # Not in the list
