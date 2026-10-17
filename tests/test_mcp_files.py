import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from dev_assistant.mcp_server import (
    MAX_FILE_BYTES,
    apply_patch,
    get_diff,
    list_files,
    read_file,
    search_text,
    write_file,
)


class MCPFileToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "one.txt").write_text("needle\nother\n", encoding="utf-8")
        (self.root / "src" / "two.txt").write_text("needle twice\n", encoding="utf-8")
        (self.root / "binary.bin").write_bytes(b"\x00\x01")
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.root)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_list_search_read_and_exact_patch(self):
        listing = list_files(self.root)
        self.assertIn("src/one.txt", listing["files"])
        self.assertEqual(read_file(self.root, "src/one.txt")["content"], "needle\nother\n")
        found = search_text(self.root, "needle")
        self.assertEqual({match["path"] for match in found["matches"]}, {"src/one.txt", "src/two.txt"})
        apply_patch(self.root, "src/one.txt", old_text="other", new_text="changed")
        self.assertIn("changed", read_file(self.root, "src/one.txt")["content"])

    def test_create_requires_missing_file_and_diff_is_scoped(self):
        write_file(self.root, "new.md", "# New\n")
        with self.assertRaises(FileExistsError):
            write_file(self.root, "new.md", "different")
        diff = get_diff(self.root, "new.md")
        self.assertEqual(diff["path"], "new.md")
        self.assertIn("new file mode", diff["diff"])
        self.assertIn("+# New", diff["diff"])

    def test_explicit_diff_includes_ignored_text_file_but_root_diff_does_not(self):
        (self.root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        ignored = self.root / "ignored" / "assistant-demo.md"
        ignored.parent.mkdir()
        ignored.write_text("# Assistant demo\n", encoding="utf-8")

        explicit = get_diff(self.root, "ignored/assistant-demo.md")
        self.assertIn("new file mode", explicit["diff"])
        self.assertIn("+# Assistant demo", explicit["diff"])

        root_diff = get_diff(self.root)
        self.assertNotIn("assistant-demo.md", root_diff["diff"])

    def test_overwrite_preserves_existing_permission_bits(self):
        target = self.root / "src" / "one.txt"
        target.chmod(0o751)
        write_file(self.root, "src/one.txt", "rewritten\n", overwrite=True)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o751)

    def test_recursive_listing_prunes_vcs_and_generated_directories(self):
        for directory in ("build", ".pytest_cache", "__pycache__", "target"):
            generated = self.root / directory
            generated.mkdir()
            (generated / "should-not-be-read.txt").write_text("secret", encoding="utf-8")
        listing = list_files(self.root, include_hidden=True)
        listed = {entry["path"] for entry in listing["entries"]}
        self.assertIn("src/one.txt", listed)
        self.assertFalse(any(path == ".git" or path.startswith((".git/", "build/", ".pytest_cache/", "__pycache__/", "target/")) for path in listed))

    def test_traversal_absolute_symlink_binary_and_size_are_rejected(self):
        for path in ("../outside", str(self.root / "src" / "one.txt")):
            with self.assertRaises(ValueError):
                read_file(self.root, path)
        outside = self.root.parent / "outside-mcp.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.root / "escape.txt").symlink_to(outside)
        with self.assertRaises(ValueError):
            read_file(self.root, "escape.txt")
        with self.assertRaises(ValueError):
            read_file(self.root, "binary.bin")
        (self.root / "large.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        with self.assertRaises(ValueError):
            read_file(self.root, "large.txt")
        outside.unlink()


if __name__ == "__main__":
    unittest.main()
