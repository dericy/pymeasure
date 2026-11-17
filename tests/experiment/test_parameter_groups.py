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

import copy
import json

import numpy as np
import pytest

from pymeasure.experiment import FloatParameter, IntegerParameter, Procedure, Results
from pymeasure.experiment.parameters import (
    ParameterGroup,
    Range1DParameterGroup,
    Range2DParameterGroup,
)
from pymeasure.experiment.procedure import ProcedureWrapper


class GroupProcedure(Procedure):
    x = FloatParameter("X", default=1.5, units="V")
    plain = ParameterGroup("Plain", a=FloatParameter("A", default=1.0),
                           b=IntegerParameter("B", default=2))
    sweep = Range1DParameterGroup("Sweep", start_kwargs={"units": "V", "default": -1})
    grid = Range2DParameterGroup("Grid", no_step_kwargs_x={"default": 3},
                                 no_step_kwargs_y={"default": 2})

    DATA_COLUMNS = ["t"]

    def execute(self):
        pass


class TestParameterGroup:
    def test_sets_group_name_of_parameters(self):
        group = ParameterGroup("Plain", a=FloatParameter("A"), b=IntegerParameter("B"))
        assert [p.group_name for p in group.parameters.values()] == ["Plain", "Plain"]
        assert group.param_names == ["a", "b"]

    def test_parameter_group_name_defaults_to_none(self):
        assert FloatParameter("A").group_name is None
        assert FloatParameter("A", group_name="G").group_name == "G"

    def test_rejects_non_parameters(self):
        with pytest.raises(TypeError, match="not a Parameter"):
            ParameterGroup("Bad", a=1.0)

    def test_groups_cannot_be_nested(self):
        with pytest.raises(TypeError, match="not a Parameter"):
            ParameterGroup("Outer", inner=ParameterGroup("Inner", a=FloatParameter("A")))

    def test_value_is_list_of_parameter_values(self):
        group = ParameterGroup("Plain", a=FloatParameter("A", default=1.0),
                               b=IntegerParameter("B", default=2))
        assert group.value == [1.0, 2]
        group.value = [3, 4]
        assert group.value == [3.0, 4]

    def test_is_set_requires_all_parameters(self):
        group = ParameterGroup("Plain", a=FloatParameter("A", default=1.0),
                               b=IntegerParameter("B"))
        assert not group.is_set()
        assert str(group) == ""
        group.parameters["b"].value = 3
        assert group.is_set()

    def test_str_is_json_with_parameter_names(self):
        group = ParameterGroup("Plain", a=FloatParameter("A", default=1.5, units="V"))
        assert json.loads(str(group)) == {"A": "1.5 V"}

    def test_deserialize_from_json_string(self):
        group = ParameterGroup("Plain", a=FloatParameter("A"), b=IntegerParameter("B"))
        group.value = json.dumps({"A": "2.5", "B": "7"})
        assert group.value == [2.5, 7]

    @pytest.mark.parametrize("group_by, expected", [
        (None, {}),
        ("enabled", {"enabled": True}),
        (["a", "b"], {"a": True, "b": True}),
        ({"a": 3}, {"a": 3}),
    ])
    def test_group_by_is_normalised_like_for_parameters(self, group_by, expected):
        assert ParameterGroup("G", group_by=group_by).group_by == expected

    def test_group_by_with_condition(self):
        assert ParameterGroup("G", group_by="mode", group_condition="fast").group_by == \
            {"mode": "fast"}

    def test_invalid_group_by(self):
        with pytest.raises(TypeError):
            ParameterGroup("G", group_by=3)


class TestRange1D:
    def test_default_values(self):
        group = Range1DParameterGroup("Sweep")
        assert group.var_name == "sweep"
        assert group.value.shape == (50,)
        assert group.value[0] == 0 and group.value[-1] == 1

    def test_specific_kwargs_override_common_ones(self):
        group = Range1DParameterGroup("Sweep", units="V", stop_kwargs={"units": "mV"})
        start, stop, steps = group.parameters.values()
        assert (start.units, stop.units) == ("V", "mV")
        assert steps.units == "V"

    def test_defaults_can_be_overridden(self):
        group = Range1DParameterGroup("Sweep", start_kwargs={"default": 5},
                                      stop_kwargs={"default": 7}, no_step_kwargs={"default": 3})
        np.testing.assert_allclose(group.value, [5, 6, 7])

    def test_deserialize_array(self):
        group = Range1DParameterGroup("Sweep")
        group.value = np.linspace(2, 4, 9)
        start, stop, steps = (p.value for p in group.parameters.values())
        assert (start, stop, steps) == (2, 4, 9)

    def test_instances_do_not_share_parameters(self):
        a, b = Range1DParameterGroup("A"), Range1DParameterGroup("B")
        a.value = np.linspace(0, 10, 5)
        assert b.value.shape == (50,)


class TestRange2D:
    def test_grid_shape_and_order(self):
        group = Range2DParameterGroup("Grid", no_step_kwargs_x={"default": 3},
                                      no_step_kwargs_y={"default": 2})
        assert group.value.shape == (6, 2)
        # x runs fastest
        np.testing.assert_allclose(group.value[:3, 0], [0, 0.5, 1])
        np.testing.assert_allclose(group.value[:, 1], [0, 0, 0, 1, 1, 1])

    def test_deserialize_points(self):
        group = Range2DParameterGroup("Grid")
        group.value = np.array([[0, 0], [1, 0], [0, 5], [1, 5]])
        values = [p.value for p in group.parameters.values()]
        assert values == [0, 1, 2, 0, 5, 2]

    def test_specific_kwargs(self):
        group = Range2DParameterGroup("Grid", units="V", start_kwargs_y={"default": -3})
        assert group.parameters["grid_start_y"].value == -3
        assert group.parameters["grid_start_x"].units == "V"


class TestGroupsInProcedure:
    def test_groups_are_collected_as_parameters(self):
        procedure = GroupProcedure()
        assert set(procedure.parameter_objects()) == {"x", "plain", "sweep", "grid"}
        assert procedure.plain == [1.0, 2]
        assert procedure.sweep.shape == (50,)
        assert procedure.sweep[0] == -1

    def test_class_attribute_is_the_group(self):
        assert isinstance(GroupProcedure.sweep, Range1DParameterGroup)

    def test_procedures_do_not_share_group_state(self):
        first, second = GroupProcedure(), GroupProcedure()
        first.sweep = np.linspace(0, 10, 5)
        first.plain = [5.0, 7]
        assert first.sweep.shape == (5,)
        assert second.sweep.shape == (50,)
        assert second.plain == [1.0, 2]
        assert GroupProcedure.sweep.value.shape == (50,)

    def test_setting_none_clears_the_group(self):
        procedure = GroupProcedure()
        procedure.plain = None
        assert procedure.plain is None
        assert not procedure.parameter_objects()["plain"].is_set()
        assert not procedure.parameters_are_set()

    def test_set_through_constructor_and_set_parameters(self):
        procedure = GroupProcedure(plain=[3.0, 4])
        assert procedure.plain == [3.0, 4]
        procedure.set_parameters({"sweep": np.linspace(1, 2, 4)})
        assert procedure.sweep.shape == (4,)

    def test_parameter_values_include_groups(self):
        values = GroupProcedure().parameter_values()
        assert set(values) == {"x", "plain", "sweep", "grid"}
        assert values["grid"].shape == (6, 2)

    def test_deepcopy_and_wrapper_pickle(self):
        procedure = GroupProcedure()
        procedure.sweep = np.linspace(2, 4, 7)
        assert copy.deepcopy(procedure.parameter_objects()["sweep"]).value.shape == (7,)
        # the wrapper sends the parameter values to the worker process
        state = ProcedureWrapper(procedure).__getstate__()
        assert state["_parameters"]["sweep"].shape == (7,)

    def test_results_round_trip(self, tmp_path):
        procedure = GroupProcedure()
        procedure.sweep = np.linspace(2, 4, 7)
        procedure.grid = np.array([[0, 0], [1, 0], [0, 5], [1, 5]])
        procedure.plain = [2.5, 9]
        filename = str(tmp_path / "groups.csv")
        Results(procedure, filename)

        loaded = Results.load(filename).procedure
        np.testing.assert_allclose(loaded.sweep, np.linspace(2, 4, 7))
        np.testing.assert_allclose(loaded.grid, [[0, 0], [1, 0], [0, 5], [1, 5]])
        assert loaded.plain == [2.5, 9]
        assert loaded.x == 1.5

    def test_placeholders_use_group_values(self, tmp_path):
        # replace_placeholders reads ``parameter.value`` of every parameter, groups included
        from pymeasure.experiment.results import replace_placeholders
        procedure = GroupProcedure()
        assert replace_placeholders("{X}", procedure) == "1.5"
