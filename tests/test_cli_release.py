import hashlib
import json

from neural_search import cli


def test_cli_release_add_pins_manifest_sha256(tmp_path, capsys):
    manifest = tmp_path / "bundle.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "example-researcher",
                "version": "1.0.0",
                "compatibility_group": "corpus:test",
                "artifacts": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.json"

    exit_code = cli.main(
        [
            "artifacts",
            "release-add",
            "--manifest",
            str(manifest),
            "--manifest-url",
            "https://example.org/releases/example-researcher-1.0.0.json",
            "--index",
            str(index),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert output["release"]["manifest_sha256"] == expected

    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["bundles"][0]["manifest_sha256"] == expected
