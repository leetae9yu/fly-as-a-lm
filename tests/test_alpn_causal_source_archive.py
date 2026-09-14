"""The source reader accepts only exact original ten-member archives."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.alpn_causal_source_archive import read_source_payloads
from tests.test_alpn_causal_calibration_worker import synthetic_context


def test_authentic_ten_member_source_contract(tmp_path: Path) -> None:
    context, _ = synthetic_context(tmp_path)
    path = context.root / "sources/seed-7-real-random_random.zip"
    assert set(read_source_payloads(path)) == {
        "checkpoint.npz",
        "report.json",
        "runtime.json",
    }
    with ZipFile(path, "a") as archive:
        archive.writestr("extra.json", b"{}")
    with pytest.raises(ValueError, match="membership"):
        _ = read_source_payloads(path)
