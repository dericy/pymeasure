#!/usr/bin/env ipython3


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

from pymeasure.instruments import Instrument, Channel
from pymeasure.instruments.validators import (
    truncated_discrete_set,
    strict_discrete_set,
    joined_validators,
    strict_range,
)
from enum import IntEnum

RANGE_OR_ZERO = joined_validators(strict_range, strict_discrete_set)


class Counter(Channel):
    class InputMode(IntEnum):
        Ten_MHz = 0
        Input_1 = 1
        Input_2 = 2
        Trig = 3

    photon_count = Channel.measurement("X{ch}", """Measure the current contents of counter""")
    input_mode = Channel.setting(
        "CI {ch},%d",
        """Set the counter to input j""",
        validator=strict_discrete_set,
        map_values=True,
        values=InputMode,
    )
    counter_preset: float = Channel.setting(
        "CP {ch},%g",
        """Set the counter {ch} to 1 <= n <= 9e11""",
        validator=strict_range,
        values=[1, 9e11],
    )
    discriminator_slope = Channel.control("DS {ch}", "DS {ch} %s", """Control the discriminator slope for {ch} to be rising edge or falling edge""",
                                         validator=strict_discrete_set,
                                         map_values=True,
                                         values = ["rise", "fall"])
    
    discriminator_mode = Channel.control("DM {ch}", "DS {ch} %s", """Control the discriminator mode for channel {ch} to be one of fixed or scan""",
                                         validator=strict_discrete_set,
                                         map_values=True,
                                         values = ["fixed", "scan"])


class SR400(Instrument):
    class CountingMode(IntEnum):
        A_AND_B = 0
        A_MINUS_B = 1
        A_PLUS_B = 2
        A_FOR_B = 3

    TIME_CONSTANTS = {
        1e-3: 1,
        3e-3: 2,
        10e-3: 3,
        30e-3: 4,
        100e-3: 5,
        300e-3: 6,
        1: 7,
        3: 8,
        10: 9,
        30: 10,
        100: 11,
    }

    SENSITIVITIES = {
        10e-9: 1,
        20e-9: 2,
        50e-9: 3,
        100e-9: 4,
        200e-9: 5,
        500e-9: 6,
        1e-6: 7,
        2e-6: 8,
        5e-6: 9,
        10e-6: 10,
        20e-6: 11,
        50e-6: 12,
        100e-6: 13,
        200e-6: 14,
        500e-6: 15,
        1e-3: 16,
        2e-3: 17,
        5e-3: 18,
        10e-3: 19,
        20e-3: 20,
        50e-3: 21,
        100e-3: 22,
        200e-3: 23,
        500e-3: 24,
    }

    counting_mode = Instrument.control(
        "CM",
        "CM %d",
        """Control the counting mode""",
        validator=strict_discrete_set,
        values=CountingMode,
        # map_values=True,
    )
    counter_periods = Instrument.setting(
        "NP %d",
        """Set the number of counter periods of counter {ch} to 1 <= n <= 2000""",
        validator=strict_range,
        values=[1, 2000],
    )

    scan_position = Instrument.measurement(
        "NN",
        """Measure the current count period number or scan position""",
        validator=strict_range,
        values=[1, 2000],
    )
    scan_end_mode = Instrument.setting(
        "NE %d",
        """Set the end of scan mode to start or stop""",
        validator=strict_range,
        values=["STOP", "START"],
    )
    dwell_time = Instrument.control(
        "DT",
        "DT %g",
        """Set the dwell time""",
        validator=RANGE_OR_ZERO,
        values=[[2e-3, 6e1], [0]],
    )
    dac_output = Instrument.setting(
        "AS %d",
        """Set the dac output to one of: {A, B, A-B, A+B}""",
        validator=strict_discrete_set,
        map_values=True,
        values=["A", "B", "A-B", "A+B"],
    )
    display_mode = Instrument.control(
        "SD",
        "SD %d",
        """Set the display mode to continuous or hold""",
        validator=strict_discrete_set,
        map_values=True,
        values=["continuous", "hold"],
    )
    gate_trigger_slope = Instrument.control(
        "TS",
        "TS %d",
        """Set the gate trigger slope to rise (positive) or fall (negative)""",
        validator=strict_discrete_set,
        map_values=True,
        values=["rise", "fall"],
    )
    gate_trigger_level = Instrument.control(
        "TL",
        "TL %d",
        """Set the gate trigger level (in Volts) to -2.000<V<2.000""",
        validator=strict_range
        values=[-2, 2],
    )
    sensitivity = Instrument.control(
        "G",
        "G%d",
        """A float property that represents the SR510 sensitivity value.
           This property can be set.""",
        validator=truncated_discrete_set,
        values=SENSITIVITIES,
        map_values=True,
    )

    frequency = Instrument.measurement(
        "F",
        """A float property representing the SR510 input reference frequency""",
    )

    status = Instrument.measurement(
        "Y",
        """A string property representing the bits set within the SR510 status byte""",
        get_process=lambda s: bin(int(s))[2:],
    )

    output = Instrument.measurement(
        "Q",
        """A float property that represents the SR510 output voltage in Volts.""",
    )

    def __init__(self, adapter, name="Stanford Research Systems SR400 Photon Counter", **kwargs):
        kwargs.setdefault("write_termination", "\r")
        super().__init__(
            adapter,
            name,
            includeSCPI=False,
            **kwargs,
        )
