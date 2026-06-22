"""CIC Assistant chat persistence — multi-turn chats + messages survive in SQLite."""
import os
import tempfile
from pathlib import Path

from storage.database import Database


def _db():
    fd, p = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Database(Path(p))


def test_chat_crud_roundtrip():
    db = _db()
    db.create_chat("c1", "First chat")
    db.add_chat_message("c1", "user", "Which zones need attention?")
    db.add_chat_message("c1", "assistant", "Zone B is at caution.")
    msgs = db.get_chat_messages("c1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Which zones need attention?"
    assert any(c["id"] == "c1" for c in db.list_chats())


def test_rename_and_delete_chat():
    db = _db()
    db.create_chat("c2", "New chat")
    db.add_chat_message("c2", "user", "hi")
    db.rename_chat("c2", "Crowd risk Q")
    assert db.list_chats()[0]["title"] == "Crowd risk Q"
    db.delete_chat("c2")
    assert db.get_chat_messages("c2") == []
    assert not db.chat_exists("c2")
