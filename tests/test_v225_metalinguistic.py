"""Source-bounded word-origin evidence is not arbitrary Latin permission."""
import hashlib

import pytest

from tarjomeh.quality.integrity import unexpected_latin_prose


@pytest.mark.parametrize("source,target,token", [
    ("The term derives from Latin civitas.", "این اصطلاح از civitas لاتین گرفته شده است.", "civitas"),
    ("The word bank comes from Italian banca.", "این واژه از banca ایتالیایی آمده است.", "banca"),
    ("The term derives from Classical Latin civitas.", "این اصطلاح از civitas آمده است.", "civitas"),
    ("The polity is a word formed on the Greek politeia – its origin is noted.",
     "این واژه از politeia یونانی ساخته شده است.", "politeia"),
])
def test_exact_source_etymon_is_not_a_latin_leak(source, target, token):
    assert not any(x["token"] == token for x in unexpected_latin_prose(source, target))


@pytest.mark.parametrize("source,target,token", [
    ("The term derives from Recent events.", "این اصطلاح events است.", "events"),
    ("The term derives from Xyzan civitas.", "این اصطلاح civitas است.", "civitas"),
    ("The name comes from Cultural traditions.", "نام traditions است.", "traditions"),
    ("The committee comes from Greek universities.", "این گروه از universities آمده است.", "universities"),
    ("The term derives from Latin ordinary prose continues here.", "این اصطلاح ordinary است.", "ordinary"),
    ("The term derives from Latin civitas.", "این اصطلاح از inventum آمده است.", "inventum"),
    ("The term derives from Latin civitas.", "civitas و civitas تکرار شده‌اند.", "civitas"),
    ("The term derives from Latin civitas.\n\nAnother subject.",
     "موضوع نخست.\n\ncivitas موضوع دیگری است.", "civitas"),
    ("The term derives from Latin civitas.\n\nAnother subject.", "این civitas است.", "civitas"),
])
def test_uncertain_or_wrong_scope_still_reports(source, target, token):
    assert any(x["token"] == token for x in unexpected_latin_prose(source, target))


def test_etymon_proof_has_exact_local_hashes_and_offsets():
    from tarjomeh.quality.integrity import source_metalinguistic_evidence

    source = "The term derives from Latin civitas."
    target = "این اصطلاح از civitas آمده است."
    [proof] = source_metalinguistic_evidence(source, target)
    assert source[proof["source_start"]:proof["source_end"]] == "civitas"
    assert target[proof["target_start"]:proof["target_end"]] == "civitas"
    assert proof["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert proof["target_sha256"] == hashlib.sha256(target.encode()).hexdigest()
    assert proof["authority"] == "source_apparatus_only_not_term_or_memory_approval"


def test_allowance_does_not_cover_adjacent_untranslated_prose():
    findings = unexpected_latin_prose(
        "The term derives from Latin civitas.", "civitas is a useful concept است.")
    assert not any(x["token"] == "civitas" for x in findings)
    assert any(x["token"] == "useful" for x in findings)


@pytest.mark.parametrize("target", [
    "civitas-like", "civitas-II", "civitas's", "civitas2", "civitas۲",
    "civitas\u200cها", "civitas\u0301",
])
def test_new_proof_never_protects_a_token_substring(target):
    from tarjomeh.quality.integrity import source_metalinguistic_evidence

    source = "The term derives from Latin civitas."
    assert source_metalinguistic_evidence(source, target) == []
    if "-" in target or "'" in target:
        assert unexpected_latin_prose(source, target)
