from mini_rlm.llm.query_functions import remove_think_tag_contents


def test_remove_think_tag_contents():
    # give: think tag is in a single line
    text = "This is a test. <think>This should be removed.</think> This should stay."
    expected = "This is a test.  This should stay."
    # when: remove_think_tag_contents is called
    # then: the content inside the think tag is removed
    assert remove_think_tag_contents(text) == expected


def test_remove_multi_line_think_tag_contents():
    # give: think tag spans multiple lines
    text = (
        "This is a test. <think>This should\n\n be\n removed.</think> This should stay."
    )
    expected = "This is a test.  This should stay."
    # when: remove_think_tag_contents is called
    # then: the content inside the think tag is removed
    assert remove_think_tag_contents(text) == expected
