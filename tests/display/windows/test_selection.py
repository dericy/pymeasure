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

from pymeasure.display.Qt import QtCore, QtGui
from pymeasure.display.widgets import ImageWidget, PlotWidget
from pymeasure.display.windows.managed_window import ManagedWindowBase
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import Range1DParameterGroup, Range2DParameterGroup


class SelectionProcedure(Procedure):
    wl = Range1DParameterGroup("Wl", start_kwargs={"default": 0}, stop_kwargs={"default": 10})
    other = Range1DParameterGroup("Other", start_kwargs={"default": 100},
                                  stop_kwargs={"default": 200})
    grid = Range2DParameterGroup("Grid", stop_kwargs_x={"default": 4},
                                 stop_kwargs_y={"default": 5})

    DATA_COLUMNS = ["x", "y", "z"]

    def execute(self):
        pass


@pytest.fixture
def window(qtbot):
    widgets = [
        PlotWidget("Plot", ["x", "y"], x_axis="x", y_axis="y", selectors=["wl", "other"]),
        ImageWidget("Image", ["x", "y", "z"], "x", "y", selectors=["grid"]),
    ]
    window = ManagedWindowBase(SelectionProcedure, inputs=["wl", "other", "grid"],
                               widget_list=widgets)
    qtbot.addWidget(window)
    window.show()
    return window


def press_a(widget):
    event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_A,
                            QtCore.Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)


def test_selectors_default_to_none(qtbot):
    assert PlotWidget("Plot", ["x", "y"]).selectors == []
    assert ImageWidget("Image", ["x", "y", "z"], "x", "y").selectors == []


def test_selectors_are_copied(qtbot):
    names = ["wl"]
    widget = PlotWidget("Plot", ["x", "y"], selectors=names)
    names.append("other")
    assert widget.selectors == ["wl"]


def test_plot_selection_round_trip(window):
    plot = window.tabs.widget(0)
    window.tabs.setCurrentIndex(0)

    window.inputs.wl.select_trailing_button.click()
    assert plot.selection_active
    assert tuple(plot.linear_reg.getRegion()) == (0, 10)  # starts from the current range

    plot.linear_reg.setRegion((2.0, 4.0))
    press_a(plot)

    assert not plot.selection_active
    assert window.inputs.wl.get_range() == (2.0, 4.0)
    assert window.inputs.other.get_range() == (100, 200)  # other selectors are untouched


def test_consecutive_selections_only_update_their_own_input(window):
    plot = window.tabs.widget(0)
    window.tabs.setCurrentIndex(0)

    window.inputs.wl.select_trailing_button.click()
    plot.linear_reg.setRegion((2.0, 4.0))
    press_a(plot)

    window.inputs.other.select_trailing_button.click()
    plot.linear_reg.setRegion((150.0, 160.0))
    press_a(plot)

    assert window.inputs.other.get_range() == (150.0, 160.0)
    # the earlier selector must not receive the result of the later selection
    assert window.inputs.wl.get_range() == (2.0, 4.0)


def test_image_selection_starts_at_the_current_range(window):
    image = window.tabs.widget(1)
    window.tabs.setCurrentIndex(1)

    window.inputs.grid.select_trailing_button.click()
    assert image.selection_active
    # without moving the rectangle the range is unchanged
    press_a(image)

    assert window.inputs.grid.get_range() == pytest.approx((0, 4, 0, 5))


def test_image_selection_round_trip(window):
    image = window.tabs.widget(1)
    window.tabs.setCurrentIndex(1)

    window.inputs.grid.select_trailing_button.click()
    image.rect_reg.setPos((1.0, 2.0))
    image.rect_reg.setSize((3.0, 4.0))
    press_a(image)

    assert window.inputs.grid.get_range() == pytest.approx((1, 4, 2, 6))


def test_selection_for_a_selector_of_another_tab_is_ignored(window):
    plot = window.tabs.widget(0)
    window.tabs.setCurrentIndex(0)
    window.inputs.grid.select_trailing_button.click()  # the plot has no "grid" selector
    assert not plot.selection_active


def test_widgets_without_selection_support_ignore_the_key(window):
    plot = window.tabs.widget(0)
    press_a(plot)  # nothing selected: must not emit or fail
    assert not plot.selection_active


def test_starting_a_selection_twice_completes_once(window):
    plot = window.tabs.widget(0)
    window.tabs.setCurrentIndex(0)
    results = []
    window.tabs.selection_completed.connect(results.append)

    window.inputs.wl.select_trailing_button.click()
    window.inputs.wl.select_trailing_button.click()  # ignored while the selection is active
    press_a(plot)

    assert len(results) == 1
