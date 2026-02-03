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
from pymeasure.instruments import Instrument, Channel, validators


class DACChannel(Channel):
    """
    Implementation of a single Basel LNHR DAC II channel.
    """

    _STEPS = 16777215  # 2^24 - 1
    enabled = Channel.control(
        "{ch} S?",
        "{ch} %s",
        """Control the output state of the channel (True/False).""",
        validator=validators.strict_discrete_set,
        values={True: "ON", False: "OFF"},
        map_values=True,
        check_set_errors=False,
    )
    voltage_setpoint = Channel.control(
        None,
        "{ch} %s",
        """Control the (DC) voltage setpoint (-10V to 10V).""",
        validator=validators.truncated_range,
        values=[-10, 10],
        set_process=lambda v: f"{int((v + 10) * 838860.74):06X}",
        check_set_errors=True,
    )

    voltage = Channel.measurement(
        "{ch} V?",
        """Measure the actual voltage in volts (float).""",
        cast=lambda v: float(int(v, 16) / (838860.74) - 10)
    )
    @staticmethod
    def _from_24bit_hex(hex_str):
        print(type(hex_str), hex_str)
        val = int(hex_str, 16)
        if val >= 0x800000:
            val -= 0x1000000
        # Dividing by 2^23 - 1 (8388607)
        return (val / 8388607) * 10
    def check_set_errors(self):
        """Mandatory handshake: DAC returns '0' on success for set commands."""
        response = self.read()
        if response != "0\r\n":
            raise RuntimeError(f"Channel Error: {response}")
        return []

    def check_errors(self):
        """Mandatory handshake: DAC returns '0' on success for set commands."""
        response = self.read()
        if response != "0\r\n":
            raise RuntimeError(f"Channel Error: {response}")
        return []


class BaspiLNHRDACII(Instrument):
    """Basel Precision Instruments Low-Noise High-Resolution DAC II with 12
    independent voltage channels.
    """
    channels = Instrument.MultiChannelCreator(DACChannel, tuple(range(1, 13)))

    def __init__(self, adapter, name="Basel LNHR DAC II", **kwargs):
        super().__init__(
            adapter,
            name,
            includeSCPI=False,
            write_termination="\n",
            read_termination="\n",
            **kwargs,
        )
    @property
    def id(self):
        """Identify the instrument. Replaces the default SCPI *IDN? behavior."""
        return self.ask("IDN?")

    @property
    def enabled_channels(self):
        """
        Get a list of the DACChannel objects that are currently enabled.

        This allows for easy iteration:
            for ch in dac.enabled_channels:
                ch.voltage_setpoint = 0
        """
        raw_response = self.ask("ALL S?")
        status_parts = [p.strip() == "ON" for p in raw_response.split(";") if p.strip()]
        return [ch for ch, is_enabled in zip(self.channels, status_parts) if is_enabled]

    def disable_all(self):
        """Turn OFF all channels that are currently enabled."""
        for ch in self.enabled_channels:
            self.channels[ch].voltage_setpoint = 0
            self.channels[ch].enabled = False

    def shutdown(self):
        """
        Sets all 12 channels to 0V and disable them.
        """
        self.disable_all()
        super().shutdown()
