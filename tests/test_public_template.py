"""Release hygiene checks; no project data required."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicTemplate(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "notebooks" / "SAA_GCR_Martin_Revision_Template.ipynb"
        self.notebook = json.loads(self.path.read_text())

    def test_notebook_has_no_executed_outputs(self):
        cells = [c for c in self.notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 9)
        for cell in cells:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
        self.assertTrue(self.notebook["metadata"]["saa_review"]["code_only_template"])

    def test_code_cells_are_plain_compilable_python(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_notebook_contains_no_private_source_links_or_attachments(self):
        text = self.path.read_text()
        for forbidden in ("drive.google.com", "/workspace/", "libfile_", "data:image/"):
            self.assertNotIn(forbidden, text)
        for cell in self.notebook["cells"]:
            self.assertFalse(cell.get("attachments"))

if __name__ == "__main__":
    unittest.main()
