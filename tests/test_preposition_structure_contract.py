"""Structural contracts for the preposition table; inspect source before dict construction."""
import ast
from pathlib import Path
import unittest


class TestPrepositionTable(unittest.TestCase):
    def test_table_keys_are_unique(self):
        source = Path(__file__).resolve().parents[1] / 'core' / 'extract.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        table = next(node.value for node in tree.body
                     if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name)
                     and node.target.id == 'PREP_CASES')
        keys = [ast.literal_eval(key) for key in table.keys]
        self.assertEqual(len(keys), len(set(keys)))
