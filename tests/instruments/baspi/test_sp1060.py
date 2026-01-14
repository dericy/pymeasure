#!/usr/bin/env python3
import pytest
from pymeasure.test import expected_protocol
from pymeasure.instruments.baspi import BaspiLNHRDACII


def test_dac_zero_volts():
    with expected_protocol(
        BaspiLNHRDACII,
        [("1 7FFFFF", "0")],  # (Sent by Python, Received from DAC)
    ) as dac:
        dac.ch_1.voltage = 0.0


def test_dac_positive_full_scale():
    """Manual: Channel 5 at 10V should send '5 FFFFFF'"""
    with expected_protocol(
        BaspiLNHRDACII,
        [("5 FFFFFF", "0")],
    ) as dac:
        dac.ch_5.voltage = 10.0


def test_dac_negative_full_scale():
    """Manual: Channel 12 at -10V should send '12 000000'"""
    with expected_protocol(
        BaspiLNHRDACII,
        [("12 000000", "0")],
    ) as dac:
        dac.ch_12.voltage = -10.0


def test_voltage_readback():
    """Verify Hex-to-Voltage conversion when reading from the DAC"""
    with expected_protocol(
        BaspiLNHRDACII,
        [("2?", "7FFFFF")],  # Query channel 2, receive midpoint hex
    ) as dac:
        assert dac.ch_2.voltage == pytest.approx(0.0, abs=1e-6)


def test_invalid_response():
    """Verify that a non-zero response triggers an error"""
    with expected_protocol(
        BaspiLNHRDACII,
        [("1 7FFFFF", "E1")],  # DAC returns error code E1
    ) as dac:
        with pytest.raises(RuntimeError):
            dac.ch_1.voltage = 0.0


def test_disable_all_logic():
    # Mocking a state where only Ch 1 and Ch 2 are ON
    with expected_protocol(
        BaselLNHRDACII,
        [
            ("ALL S?", "ON;ON;OFF;OFF;OFF;OFF;OFF;OFF;OFF;OFF;OFF;OFF"),
            ("1 OFF", "0"),
            ("2 OFF", "0"),
        ],
    ) as dac:
        dac.disable_all()
