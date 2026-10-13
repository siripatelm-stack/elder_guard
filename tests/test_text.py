from elder_guard.text import split_turns


def test_speaker_markers():
    assert split_turns("caller: hi there receiver: hello") == ["caller: hi there", "receiver: hello"]


def test_sentence_fallback():
    assert split_turns("Hello there. Send money now!") == ["Hello there.", "Send money now!"]


def test_empty():
    assert split_turns("   ") == []
