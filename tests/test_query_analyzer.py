from src.search.query_analyzer import rule_based_analyze


def test_rule_analyzer_video_search():
    plan = rule_based_analyze("\uae40\uce58\ucc0c\uac1c \uc601\uc0c1 \ucc3e\uc544\uc918")
    assert plan.intent == "video_search"
    assert plan.query_type == "recipe"
    assert plan.weights.text > plan.weights.image


def test_rule_analyzer_scene_search():
    plan = rule_based_analyze("\ub300\ud30c \ub123\ub294 \uc7a5\uba74")
    assert plan.intent == "scene_search"
    assert plan.scene_query
    assert plan.weights.text > plan.weights.image


def test_rule_analyzer_compound_scene_search():
    plan = rule_based_analyze("\uae40\uce58\ucc0c\uac1c \uc601\uc0c1\uc5d0\uc11c \ub300\ud30c \ub123\ub294 \uc7a5\uba74 \ubcf4\uc5ec\uc918")
    assert plan.intent == "compound_scene_search"
    assert plan.scope == "video_candidate"
    assert "\uae40\uce58\ucc0c\uac1c" in plan.video_query
    assert "\ub300\ud30c" in plan.scene_query


def test_rule_analyzer_summary():
    plan = rule_based_analyze("\uc774 \uc601\uc0c1 \uc7ac\ub8cc \uc815\ub9ac\ud574\uc918", optional_video_id="short_001")
    assert plan.intent == "summary"
    assert plan.needs_generation is True
    assert plan.scope == "video_id"
