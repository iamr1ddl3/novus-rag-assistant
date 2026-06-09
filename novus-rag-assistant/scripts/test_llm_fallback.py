"""
test_llm_fallback.py — Gap 2 unit tests for the shared fallback helper.

Run from the project root:
    python -m pytest scripts/test_llm_fallback.py -v
    # or, no pytest:
    python scripts/test_llm_fallback.py

Tests the raw-OpenAI path (LITELLM_ENABLED=False), since that is the branch
that previously had NO fallback. The LiteLLM path delegates fallback to litellm
itself and is covered by litellm's own tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.llm_fallback as lf


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_fake_client(fail_models):
    """Return a fake OpenAI client that raises for models in `fail_models`,
    else echoes the model name it was called with."""
    calls = []

    class _Completions:
        def create(self, *, model, **kwargs):
            calls.append(model)
            if model in fail_models:
                raise RuntimeError(f"simulated outage for {model}")
            return _FakeResponse(f"ok:{model}")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client(), calls


def test_primary_succeeds(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client(fail_models=set())
    monkeypatch.setattr(lf, "_client", client)

    out = lf.llm_complete([{"role": "user", "content": "hi"}])
    assert out == f"ok:{lf.PRIMARY_MODEL}"
    assert calls == [lf.PRIMARY_MODEL]  # fallback never invoked


def test_falls_back_on_primary_outage(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client(fail_models={lf.PRIMARY_MODEL})
    monkeypatch.setattr(lf, "_client", client)

    out = lf.llm_complete([{"role": "user", "content": "hi"}])
    assert out == f"ok:{lf.FALLBACK_MODEL}"          # returned via fallback
    assert calls == [lf.PRIMARY_MODEL, lf.FALLBACK_MODEL]  # tried primary first


def test_raises_when_both_models_down(monkeypatch):
    monkeypatch.setattr(lf, "LITELLM_ENABLED", False)
    client, calls = _make_fake_client(fail_models={lf.PRIMARY_MODEL, lf.FALLBACK_MODEL})
    monkeypatch.setattr(lf, "_client", client)

    try:
        lf.llm_complete([{"role": "user", "content": "hi"}])
        assert False, "expected RuntimeError when both models fail"
    except RuntimeError:
        pass
    assert calls == [lf.PRIMARY_MODEL, lf.FALLBACK_MODEL]


if __name__ == "__main__":
    # Minimal no-pytest runner with a tiny monkeypatch shim.
    class _MP:
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, val in reversed(self._undo): setattr(obj, name, val)
            self._undo = []

    for fn in [test_primary_succeeds, test_falls_back_on_primary_outage, test_raises_when_both_models_down]:
        mp = _MP()
        try:
            fn(mp)
            print(f"  PASS  {fn.__name__}")
        finally:
            mp.undo()
    print("✅ all fallback tests passed")
