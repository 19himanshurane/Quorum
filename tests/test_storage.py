from arbitration.graph import run_arbitration
from arbitration.storage import count_arbitrations, get_arbitration, list_arbitrations, save_arbitration


def test_save_and_retrieve_round_trip(tmp_db_path):
    record = run_arbitration("The Eiffel Tower is in London.", "Where is the Eiffel Tower?")
    save_arbitration(tmp_db_path, record)

    fetched = get_arbitration(tmp_db_path, record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.verdict.overall_score == record.verdict.overall_score
    assert fetched.original_output == record.original_output


def test_get_missing_returns_none(tmp_db_path):
    assert get_arbitration(tmp_db_path, "nonexistent") is None


def test_list_and_count(tmp_db_path):
    for text in ["output one", "output two", "output three"]:
        save_arbitration(tmp_db_path, run_arbitration(text))

    assert count_arbitrations(tmp_db_path) == 3
    assert len(list_arbitrations(tmp_db_path, limit=2)) == 2
    assert len(list_arbitrations(tmp_db_path, limit=10)) == 3
