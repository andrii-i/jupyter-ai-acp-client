from acp.schema import FileEditToolCallContent

from jupyter_ai_acp_client.tool_call_renderer import (
    ToolCallDiff,
    ToolCallState,
    _shorten_title,
    _generate_title,
    _parse_unified_diff,
    extract_diffs,
    extract_diffs_from_raw_input,
    update_tool_call_from_start,
    update_tool_call_from_progress,
)


def _serialize(tool_calls: dict[str, ToolCallState]) -> list[dict]:
    """Helper to serialize tool calls using model_dump."""
    return [tc.model_dump(exclude_none=True) for tc in tool_calls.values()]


class TestUpdateToolCallFromStart:
    def test_creates_new_tool_call(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Reading file.py...",
            kind="read",
        )
        assert "tc-1" in tool_calls
        tc = tool_calls["tc-1"]
        assert tc.tool_call_id == "tc-1"
        assert tc.title == "Reading file.py..."
        assert tc.kind == "read"
        assert tc.status == "in_progress"
        assert tc.raw_output is None

    def test_overwrites_existing_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Old title",
                kind="read",
                status="completed",
            )
        }
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="New title",
            kind="edit",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "New title"
        assert tc.kind == "edit"
        assert tc.status == "completed"  # status preserved on merge

    def test_empty_title_generates_from_kind_and_locations(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="",
            kind="read",
            locations=["/Users/foo/project/justfile"],
        )
        assert tool_calls["tc-1"].title == "Reading justfile"


class TestUpdateToolCallFromProgress:
    def test_updates_existing_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read file.py (42 lines)",
            status="completed",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "Read file.py (42 lines)"
        assert tc.status == "completed"

    def test_creates_if_not_exists(self):
        tool_calls = {}
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Some progress",
            status="in_progress",
        )
        assert "tc-1" in tool_calls
        tc = tool_calls["tc-1"]
        assert tc.title == "Some progress"
        assert tc.status == "in_progress"

    def test_partial_update_preserves_existing(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        # Only update status, not title
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            status="completed",
        )
        tc = tool_calls["tc-1"]
        assert tc.title == "Reading file.py..."  # preserved
        assert tc.status == "completed"  # updated


class TestSerializeToolCalls:
    def test_single_tool_call(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Reading file.py...",
                kind="read",
                status="in_progress",
            )
        }
        result = _serialize(tool_calls)
        assert len(result) == 1
        assert result[0] == {
            "tool_call_id": "tc-1",
            "title": "Reading file.py...",
            "kind": "read",
            "status": "in_progress",
        }

    def test_serializes_diffs(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Editing b.py",
                kind="edit",
                status="completed",
                diffs=[ToolCallDiff(path="/a/b.py", new_text="new", old_text="old")],
            )
        }
        result = _serialize(tool_calls)
        assert result[0]["diffs"] == [
            {"path": "/a/b.py", "new_text": "new", "old_text": "old"}
        ]


class TestShortenTitle:
    def test_absolute_path(self):
        assert _shorten_title("Read /Users/foo/bar/justfile") == "Read justfile"

    def test_multiple_absolute_paths(self):
        assert _shorten_title("Moved /a/b/old.py /a/b/new.py") == "Moved old.py new.py"

    def test_no_paths(self):
        assert _shorten_title("Read File") == "Read File"

    def test_relative_path(self):
        # Doesn't start with / — not an absolute path, leave as-is
        assert _shorten_title("Read src/foo.py") == "Read src/foo.py"

    def test_single_component_path(self):
        # Starts with / but only one component — not shortened
        assert _shorten_title("Read /justfile") == "Read /justfile"

    def test_only_path(self):
        assert _shorten_title("/Users/foo/bar/baz.py") == "baz.py"


class TestShortenTitleIntegration:
    def test_start_shortens_agent_title(self):
        tool_calls = {}
        update_tool_call_from_start(
            tool_calls,
            tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            kind="read",
        )
        assert tool_calls["tc-1"].title == "Read justfile"

    def test_progress_shortens_title_update(self):
        tool_calls = {
            "tc-1": ToolCallState(
                tool_call_id="tc-1",
                title="Read File",
                kind="read",
                status="in_progress",
            )
        }
        update_tool_call_from_progress(
            tool_calls,
            tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            status="completed",
        )
        assert tool_calls["tc-1"].title == "Read justfile"

    def test_full_flow_start_start_progress(self):
        """Simulate actual agent flow: two starts + completed progress."""
        tool_calls = {}

        # ToolCallStart #1: generic title, no locations
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Read File", kind="read",
        )
        assert tool_calls["tc-1"].title == "Read File"
        assert tool_calls["tc-1"].status == "in_progress"

        # ToolCallStart #2: full path, with locations
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Read /Users/aieroshe/Documents/project/justfile",
            kind="read",
            locations=["/Users/aieroshe/Documents/project/justfile"],
        )
        assert tool_calls["tc-1"].title == "Read justfile"
        assert tool_calls["tc-1"].locations == ["/Users/aieroshe/Documents/project/justfile"]

        # ToolCallProgress: completed, no title change
        update_tool_call_from_progress(
            tool_calls, tool_call_id="tc-1",
            status="completed",
        )
        assert tool_calls["tc-1"].title == "Read justfile"
        assert tool_calls["tc-1"].status == "completed"

        # Serialize — locations should be included for frontend
        result = _serialize(tool_calls)
        assert result[0]["title"] == "Read justfile"
        assert result[0]["locations"] == ["/Users/aieroshe/Documents/project/justfile"]
        assert result[0]["status"] == "completed"

    def test_full_flow_with_diffs(self):
        """Simulate edit flow: start → start-with-diffs → progress-completed."""
        tool_calls = {}
        diffs = [ToolCallDiff(path="/a/b.py", new_text="new code", old_text="old code")]

        # ToolCallStart #1: no diffs yet
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Editing b.py", kind="edit",
        )
        assert tool_calls["tc-1"].diffs is None

        # ToolCallStart #2: diffs arrive
        update_tool_call_from_start(
            tool_calls, tool_call_id="tc-1",
            title="Editing b.py", kind="edit",
            diffs=diffs,
        )
        assert tool_calls["tc-1"].diffs == diffs

        # ToolCallProgress: completed, diffs preserved
        update_tool_call_from_progress(
            tool_calls, tool_call_id="tc-1",
            status="completed",
        )
        assert tool_calls["tc-1"].diffs == diffs
        assert tool_calls["tc-1"].status == "completed"

        # Serialize — diffs should be included for frontend
        result = _serialize(tool_calls)
        assert result[0]["diffs"] == [
            {"path": "/a/b.py", "new_text": "new code", "old_text": "old code"}
        ]


class TestGenerateTitle:
    def test_kind_with_locations(self):
        assert _generate_title("read", ["/Users/foo/project/justfile"]) == "Reading justfile"

    def test_kind_without_locations(self):
        assert _generate_title("read") == "Reading..."

    def test_execute_kind(self):
        assert _generate_title("execute") == "Running command..."

    def test_no_kind_no_locations(self):
        assert _generate_title(None) == "Working..."

    def test_location_without_slash(self):
        assert _generate_title("read", ["justfile"]) == "Reading justfile"

    def test_multiple_locations_uses_first(self):
        assert _generate_title("read", ["/a/b/first.py", "/a/b/second.py"]) == "Reading first.py"


class TestExtractDiffs:
    def test_extracts_file_edit_content(self):
        content = [
            FileEditToolCallContent(
                path="/a/b.py", newText="new", oldText="old", type="diff"
            )
        ]
        result = extract_diffs(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].path == "/a/b.py"
        assert result[0].new_text == "new"
        assert result[0].old_text == "old"

    def test_returns_none_for_empty_content(self):
        assert extract_diffs([]) is None

    def test_skips_non_file_edit_items(self):
        content = [
            "not a FileEditToolCallContent",
            FileEditToolCallContent(
                path="/a/b.py", newText="new", oldText="old", type="diff"
            ),
            42,
        ]
        result = extract_diffs(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].path == "/a/b.py"

    def test_new_file_has_none_old_text(self):
        content = [
            FileEditToolCallContent(
                path="/a/new.py", newText="hello", type="diff"
            )
        ]
        result = extract_diffs(content)
        assert result is not None
        assert result[0].old_text is None

    def test_relative_path_normalized_with_root_dir(self):
        content = [
            FileEditToolCallContent(
                path="b.py", newText="new", type="diff"
            )
        ]
        result = extract_diffs(content, root_dir="/srv")
        assert result is not None
        assert result[0].path == "/srv/b.py"

    def test_absolute_path_unchanged_with_root_dir(self):
        content = [
            FileEditToolCallContent(
                path="/a/b.py", newText="new", type="diff"
            )
        ]
        result = extract_diffs(content, root_dir="/srv")
        assert result is not None
        assert result[0].path == "/a/b.py"

    def test_no_root_dir_preserves_relative_path(self):
        content = [
            FileEditToolCallContent(
                path="b.py", newText="new", type="diff"
            )
        ]
        result = extract_diffs(content)
        assert result is not None
        assert result[0].path == "b.py"

    def test_tilde_path_expanded_with_root_dir(self):
        content = [
            FileEditToolCallContent(
                path="~/docs/b.py", newText="new", type="diff"
            )
        ]
        result = extract_diffs(content, root_dir="/srv")
        assert result is not None
        # expanduser() resolves ~ to actual home dir
        assert result[0].path.endswith("/docs/b.py")
        assert not result[0].path.startswith("~")
        assert result[0].path.startswith("/")

    def test_dotdot_path_resolved_with_root_dir(self):
        content = [
            FileEditToolCallContent(
                path="../nbs/b.py", newText="new", type="diff"
            )
        ]
        result = extract_diffs(content, root_dir="/srv/root")
        assert result is not None
        assert result[0].path == "/srv/nbs/b.py"



class TestParseUnifiedDiff:
    def test_new_file(self):
        diff = (
            "Index: foo.py\n"
            "===================================================================\n"
            "--- foo.py\n"
            "+++ foo.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+print('hello')\n"
        )
        result = _parse_unified_diff(diff)
        assert result is not None
        old, new = result
        assert old == ""
        assert new == "print('hello')"

    def test_edit_file(self):
        diff = (
            "Index: foo.py\n"
            "===================================================================\n"
            "--- foo.py\n"
            "+++ foo.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+# comment\n"
            " print('hello')\n"
        )
        result = _parse_unified_diff(diff)
        assert result is not None
        old, new = result
        assert old == "print('hello')"
        assert new == "# comment\nprint('hello')"

    def test_delete_lines(self):
        diff = (
            "@@ -1,2 +1,1 @@\n"
            "-# old comment\n"
            " print('hello')\n"
        )
        result = _parse_unified_diff(diff)
        assert result is not None
        old, new = result
        assert old == "# old comment\nprint('hello')"
        assert new == "print('hello')"

    def test_multiple_hunks(self):
        diff = (
            "@@ -1,1 +1,1 @@\n"
            "-old_a\n"
            "+new_a\n"
            "@@ -10,1 +10,1 @@\n"
            "-old_b\n"
            "+new_b\n"
        )
        result = _parse_unified_diff(diff)
        assert result is not None
        old, new = result
        assert old == "old_a\nold_b"
        assert new == "new_a\nnew_b"

    def test_not_a_diff(self):
        assert _parse_unified_diff("just a string") is None

    def test_empty_hunks(self):
        assert _parse_unified_diff("@@ -0,0 +0,0 @@\n") is None


class TestExtractDiffsFromRawInput:
    def test_filepath_and_diff(self):
        raw = {
            "filepath": "/tmp/foo.py",
            "diff": "@@ -0,0 +1,1 @@\n+hello\n",
        }
        result = extract_diffs_from_raw_input(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0].path == "/tmp/foo.py"
        assert result[0].new_text == "hello"
        assert result[0].old_text is None

    def test_camelCase_filePath(self):
        raw = {
            "filePath": "/tmp/foo.py",
            "diff": "@@ -0,0 +1,1 @@\n+hello\n",
        }
        result = extract_diffs_from_raw_input(raw)
        assert result is not None
        assert result[0].path == "/tmp/foo.py"

    def test_edit_preserves_old_text(self):
        raw = {
            "filepath": "/tmp/foo.py",
            "diff": "@@ -1,1 +1,2 @@\n+# new\n old\n",
        }
        result = extract_diffs_from_raw_input(raw)
        assert result is not None
        assert result[0].old_text == "old"
        assert result[0].new_text == "# new\nold"

    def test_relative_path_resolved(self):
        raw = {
            "filepath": "foo.py",
            "diff": "@@ -0,0 +1,1 @@\n+x\n",
        }
        result = extract_diffs_from_raw_input(raw, root_dir="/srv/project")
        assert result is not None
        assert result[0].path == "/srv/project/foo.py"

    def test_absolute_path_unchanged(self):
        raw = {
            "filepath": "/abs/path/foo.py",
            "diff": "@@ -0,0 +1,1 @@\n+x\n",
        }
        result = extract_diffs_from_raw_input(raw, root_dir="/home/user/project")
        assert result is not None
        assert result[0].path == "/abs/path/foo.py"
