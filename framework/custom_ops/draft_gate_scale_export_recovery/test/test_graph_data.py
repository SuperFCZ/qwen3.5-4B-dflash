import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_data import compare_for_diagnosis, const_mapping, unorder_attrs


def fixture(name):
    return {
        name: {"name": name, "type": "Const", "attr": [{"key": "value", "value": {"t": {"data": "Af8="}}}]},
        "use": {"name": "use", "type": "Add", "input": ["x:0", name + ":0"]},
    }


class MappingTests(unittest.TestCase):
    def test_generated_name_at_same_port(self):
        a, b = fixture("dynamic_const_100_3"), fixture("dynamic_const_300_7")
        aliases, missing, count, _ = compare_for_diagnosis(a, b)
        self.assertEqual(aliases, {"dynamic_const_300_7": "dynamic_const_100_3"})
        self.assertEqual(missing, [[], []])
        self.assertFalse(count)
        b["dynamic_const_300_7"]["attr"][0]["value"]["t"]["data"] = "Bf8="
        self.assertTrue(compare_for_diagnosis(a, b)[2])

    def test_changed_port_and_disconnected_node_remain_unpaired(self):
        a, b = fixture("dynamic_const_100_3"), fixture("dynamic_const_300_7")
        b["use"]["input"].reverse()
        self.assertFalse(const_mapping(a, b)[0])
        b["use"]["input"] = []
        self.assertTrue(const_mapping(a, b)[1][1])

    def test_nongenerated_rename_and_attribute_multiplicity(self):
        a, b = fixture("frozen_const"), fixture("another_const")
        self.assertFalse(const_mapping(a, b)[0])
        attrs = [{"key": "fmt", "value": {"i": 0}}, {"key": "fmt", "value": {"i": 2}}]
        self.assertEqual(unorder_attrs(attrs, "attr"), unorder_attrs(attrs[::-1], "attr"))
        self.assertNotEqual(unorder_attrs(attrs, "attr"), unorder_attrs(attrs[:1], "attr"))
