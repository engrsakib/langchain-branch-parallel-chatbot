
import pytest

from app.utils.helpers import (
    format_confidence,
    format_keyword_badges,
    sanitize_query,
    state_list,
)


class TestSanitizeQuery:
    def test_trims_the_edges(self):
        assert sanitize_query("  hello  ") == "hello"

    def test_preserves_indentation_inside_pasted_code(self):
        snippet = "def f():\n    return 1"
        assert sanitize_query(snippet) == snippet

    def test_normalises_windows_line_endings(self):
        assert sanitize_query("a\r\nb") == "a\nb"

    def test_removes_control_characters(self):
        assert sanitize_query("he\x00ll\x07o") == "hello"

    def test_keeps_tabs_and_newlines(self):
        assert sanitize_query("a\tb\nc") == "a\tb\nc"

    def test_strips_trailing_spaces_from_each_line(self):
        assert sanitize_query("a   \nb  ") == "a\nb"

    def test_collapses_long_runs_of_blank_lines(self):
        assert sanitize_query("a\n\n\n\n\nb") == "a\n\nb"

    def test_truncates_to_the_requested_length(self):
        assert sanitize_query("x" * 20, max_length=5) == "xxxxx"

    def test_does_not_pad_short_input(self):
        assert sanitize_query("hi", max_length=100) == "hi"

    @pytest.mark.parametrize("value", [None, 42, ["a"]])
    def test_non_strings_become_empty(self, value):
        assert sanitize_query(value) == ""


class TestFormatKeywordBadges:
    def test_renders_one_badge_per_keyword(self):
        assert format_keyword_badges(["a", "b"]) == ":gray-background[a] :gray-background[b]"

    def test_honours_the_colour(self):
        assert format_keyword_badges(["a"], color="blue") == ":blue-background[a]"

    def test_drops_brackets_that_would_break_the_directive(self):
        assert format_keyword_badges(["li[st]s"]) == ":gray-background[lists]"

    def test_skips_blank_keywords(self):
        assert format_keyword_badges(["", "  ", "kept"]) == ":gray-background[kept]"

    def test_empty_input_produces_empty_output(self):
        assert format_keyword_badges([]) == ""


class TestFormatConfidence:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, "0%"), (0.5, "50%"), (0.826, "83%"), (1.0, "100%")],
    )
    def test_formats_as_a_whole_percentage(self, value, expected):
        assert format_confidence(value) == expected


class Item:
    pass


class TestStateList:
    def test_creates_a_missing_key(self):
        state = {}
        assert state_list(state, "items", Item) == []
        assert state["items"] == []

    def test_returns_matching_items_untouched(self):
        item = Item()
        state = {"items": [item]}
        assert state_list(state, "items", Item) == [item]

    def test_discards_entries_of_the_wrong_type(self):
                                                                                 
        item = Item()
        state = {"items": [item, "not an item", 7]}
        assert state_list(state, "items", Item) == [item]
        assert state["items"] == [item]

    def test_replaces_a_value_that_is_not_a_list(self):
        state = {"items": "corrupted"}
        assert state_list(state, "items", Item) == []
        assert state["items"] == []

    def test_returns_the_live_list_so_appends_persist(self):
        state = {}
        returned = state_list(state, "items", Item)
        returned.append(Item())
        assert len(state["items"]) == 1
