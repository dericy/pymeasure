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

import numpy as np
import pytest

from pymeasure.display.input_groups import InputGroup, Range1DInputGroup, Range2DInputGroup
from pymeasure.display.inputs import IntegerInput, ScientificInput
from pymeasure.display.widgets import InputsWidget
from pymeasure.experiment import FloatParameter, IntegerParameter, Procedure
from pymeasure.experiment.parameters import (
    ParameterGroup,
    Range1DParameterGroup,
    Range2DParameterGroup,
)


class GroupProcedure(Procedure):
    x = FloatParameter("X", default=1.0)
    plain = ParameterGroup("Plain", a=FloatParameter("A", default=1.0),
                           b=IntegerParameter("B", default=2))
    sweep = Range1DParameterGroup("Sweep")
    grid = Range2DParameterGroup("Grid")

    DATA_COLUMNS = ["t"]

    def execute(self):
        pass


@pytest.fixture
def inputs(qtbot):
    widget = InputsWidget(GroupProcedure, inputs=["x", "plain", "sweep", "grid"])
    qtbot.addWidget(widget)
    return widget


def test_widget_classes_for_groups(inputs):
    # The range groups are subclasses of ParameterGroup and need their own input group
    assert type(inputs.plain) is InputGroup
    assert type(inputs.sweep) is Range1DInputGroup
    assert type(inputs.grid) is Range2DInputGroup
    assert isinstance(inputs.x, ScientificInput)


def test_group_contains_inputs_for_its_parameters(inputs):
    assert list(inputs.plain.elements) == ["A", "B"]
    assert isinstance(inputs.plain.elements["A"], ScientificInput)
    assert isinstance(inputs.plain.elements["B"], IntegerInput)
    assert inputs.plain.elements["A"].get_value() == 1.0


def test_gui_edits_reach_the_procedure(inputs):
    """Regression test: the values typed in a group must end up in the procedure."""
    inputs.sweep.elements["Start"].set_value(10.0)
    inputs.sweep.elements["Stop"].set_value(20.0)
    inputs.sweep.elements["Number of steps"].set_value(3)
    inputs.plain.elements["A"].set_value(7.5)
    inputs.x.set_value(2.5)

    procedure = inputs.get_procedure()

    assert procedure.x == 2.5
    np.testing.assert_allclose(procedure.sweep, [10, 15, 20])
    assert procedure.plain == [7.5, 2]


def test_reading_parameter_updates_from_the_gui(inputs):
    inputs.plain.elements["B"].set_value(5)
    assert inputs.plain.parameter.parameters["b"].value == 5


def test_set_parameters_loads_values_into_groups(inputs):
    procedure = GroupProcedure()
    procedure.sweep = np.linspace(2, 4, 9)
    procedure.plain = [3.5, 8]
    procedure.grid = np.array([[0, 0], [1, 0], [0, 5], [1, 5]])

    inputs.set_parameters(procedure.parameter_objects())

    assert inputs.sweep.elements["Start"].get_value() == 2
    assert inputs.sweep.elements["Number of steps"].get_value() == 9
    assert inputs.plain.elements["A"].get_value() == 3.5
    assert inputs.grid.elements["Stop y"].get_value() == 5
    np.testing.assert_allclose(inputs.get_procedure().sweep, np.linspace(2, 4, 9))


def test_reset_restores_defaults(inputs):
    element = inputs.plain.elements["A"]
    element.set_value(9.0)
    element.reset()
    assert element.get_value() == 1.0


def test_range_1d_selection(inputs, qtbot):
    inputs.sweep.elements["Start"].set_value(1.0)
    inputs.sweep.elements["Stop"].set_value(3.0)
    assert inputs.sweep.get_range() == (1.0, 3.0)

    with qtbot.waitSignal(inputs.sweep.selection_triggered) as blocker:
        inputs.sweep.select_trailing_button.click()
    assert blocker.args == [(1.0, 3.0)]

    inputs.sweep.set_range((5.0, 6.0))
    assert inputs.sweep.get_range() == (5.0, 6.0)


def test_range_2d_selection(inputs, qtbot):
    inputs.grid.set_range((1.0, 2.0, 3.0, 4.0))
    assert inputs.grid.get_range() == (1.0, 2.0, 3.0, 4.0)
    with qtbot.waitSignal(inputs.grid.selection_triggered) as blocker:
        inputs.grid.select_trailing_button.click()
    assert blocker.args == [(1.0, 2.0, 3.0, 4.0)]


def test_inputs_widget_forwards_selection_with_dimension_tag(inputs, qtbot):
    with qtbot.waitSignal(inputs.selection_triggered) as blocker:
        inputs.sweep.select_trailing_button.click()
    assert blocker.args[:2] == ["sweep", "1D"]
    with qtbot.waitSignal(inputs.selection_triggered) as blocker:
        inputs.grid.select_trailing_button.click()
    assert blocker.args[:2] == ["grid", "2D"]


def test_set_selector_updates_the_group(inputs):
    inputs.set_selector(["sweep", (2.0, 4.0)])
    assert inputs.sweep.get_range() == (2.0, 4.0)
