"""Tests for neural_search.ingestion.doi_utils."""

from __future__ import annotations

from neural_search.ingestion.doi_utils import (
    dois_to_paper_ids,
    extract_dois_from_dandi_metadata,
    extract_dois_from_openneuro_metadata,
    extract_dois_from_text,
)


class TestExtractDoisFromText:
    def test_bare_doi_in_text(self) -> None:
        text = "See paper 10.1038/s41593-019-0409-2 for details."
        result = extract_dois_from_text(text)
        assert result == ["10.1038/s41593-019-0409-2"]

    def test_doi_org_url(self) -> None:
        text = "Published at https://doi.org/10.1234/journal.abc.001."
        result = extract_dois_from_text(text)
        assert result == ["10.1234/journal.abc.001"]

    def test_doi_colon_prefix(self) -> None:
        # The regex matches the bare 10.xxx form directly in text
        text = "Reference: doi:10.5555/test-2023"
        result = extract_dois_from_text(text)
        assert "10.5555/test-2023" in result

    def test_multiple_dois_deduped(self) -> None:
        text = (
            "10.1038/abc and again 10.1038/abc plus new 10.9999/xyz"
        )
        result = extract_dois_from_text(text)
        assert result.count("10.1038/abc") == 1
        assert "10.9999/xyz" in result
        assert len(result) == 2

    def test_max_10_results(self) -> None:
        # Build a text with 12 distinct DOIs
        dois = [f"10.1234/paper{i:02d}" for i in range(12)]
        text = " ".join(dois)
        result = extract_dois_from_text(text)
        assert len(result) == 10

    def test_no_dois_returns_empty(self) -> None:
        assert extract_dois_from_text("No DOIs here at all.") == []

    def test_empty_string_returns_empty(self) -> None:
        assert extract_dois_from_text("") == []

    def test_strips_trailing_period(self) -> None:
        text = "Cite as 10.1038/nature12345."
        result = extract_dois_from_text(text)
        assert result == ["10.1038/nature12345"]

    def test_strips_trailing_comma(self) -> None:
        text = "First ref: 10.1016/j.cell.2020.01.001, second ref follows"
        result = extract_dois_from_text(text)
        assert "10.1016/j.cell.2020.01.001" in result

    def test_https_doi_url_extracted_as_bare(self) -> None:
        text = "Published at https://doi.org/10.7554/eLife.60606"
        result = extract_dois_from_text(text)
        assert "10.7554/eLife.60606" in result


class TestExtractDoisFromDandiMetadata:
    def test_related_resource_doi_url(self) -> None:
        metadata = {
            "relatedResource": [
                {
                    "relation": "isDescribedBy",
                    "url": "https://doi.org/10.1038/s41593-019-0409-2",
                    "schemaKey": "Resource",
                }
            ]
        }
        result = extract_dois_from_dandi_metadata(metadata)
        assert result == ["10.1038/s41593-019-0409-2"]

    def test_direct_doi_field(self) -> None:
        metadata = {"doi": "10.48324/dandi.000001.0.230101.1548"}
        result = extract_dois_from_dandi_metadata(metadata)
        assert "10.48324/dandi.000001.0.230101.1548" in result

    def test_direct_doi_field_as_url(self) -> None:
        metadata = {"doi": "https://doi.org/10.48324/dandi.000002"}
        result = extract_dois_from_dandi_metadata(metadata)
        assert "10.48324/dandi.000002" in result

    def test_url_field_with_doi_org(self) -> None:
        metadata = {"url": "https://doi.org/10.5281/zenodo.999999"}
        result = extract_dois_from_dandi_metadata(metadata)
        assert "10.5281/zenodo.999999" in result

    def test_description_fallback(self) -> None:
        metadata = {
            "description": (
                "This dataset accompanies 10.1101/2021.01.01.000001 "
                "and the companion analysis."
            )
        }
        result = extract_dois_from_dandi_metadata(metadata)
        assert "10.1101/2021.01.01.000001" in result

    def test_multiple_related_resources_deduped(self) -> None:
        metadata = {
            "relatedResource": [
                {"url": "https://doi.org/10.1038/abc"},
                {"url": "https://doi.org/10.1038/abc"},
                {"url": "https://doi.org/10.9999/xyz"},
            ]
        }
        result = extract_dois_from_dandi_metadata(metadata)
        assert result.count("10.1038/abc") == 1
        assert "10.9999/xyz" in result
        assert len(result) == 2

    def test_empty_metadata_returns_empty(self) -> None:
        assert extract_dois_from_dandi_metadata({}) == []

    def test_related_resource_with_no_url(self) -> None:
        metadata = {
            "relatedResource": [
                {"relation": "isSupplementTo", "schemaKey": "Resource"}
            ]
        }
        assert extract_dois_from_dandi_metadata(metadata) == []

    def test_none_related_resource_ignored(self) -> None:
        metadata = {"relatedResource": None}
        result = extract_dois_from_dandi_metadata(metadata)
        assert result == []


class TestExtractDoisFromOpenNeuroMetadata:
    def test_references_and_links_with_doi_url(self) -> None:
        metadata = {
            "dataset": {
                "description": {
                    "ReferencesAndLinks": [
                        "https://doi.org/10.1016/j.neuroimage.2020.116946"
                    ]
                }
            }
        }
        result = extract_dois_from_openneuro_metadata(metadata)
        assert "10.1016/j.neuroimage.2020.116946" in result

    def test_top_level_doi_field(self) -> None:
        metadata = {"doi": "10.18112/openneuro.ds003505.v1.1.0"}
        result = extract_dois_from_openneuro_metadata(metadata)
        assert "10.18112/openneuro.ds003505.v1.1.0" in result

    def test_top_level_doi_as_url(self) -> None:
        metadata = {"doi": "https://doi.org/10.18112/openneuro.ds000001.v1"}
        result = extract_dois_from_openneuro_metadata(metadata)
        assert "10.18112/openneuro.ds000001.v1" in result

    def test_flat_description_references_and_links(self) -> None:
        metadata = {
            "description": {
                "ReferencesAndLinks": [
                    "https://doi.org/10.1038/s41586-021-03560-w"
                ]
            }
        }
        result = extract_dois_from_openneuro_metadata(metadata)
        assert "10.1038/s41586-021-03560-w" in result

    def test_description_string_fallback(self) -> None:
        metadata = {
            "description": "Based on 10.1101/2022.01.28.478126"
        }
        result = extract_dois_from_openneuro_metadata(metadata)
        assert "10.1101/2022.01.28.478126" in result

    def test_empty_metadata_returns_empty(self) -> None:
        assert extract_dois_from_openneuro_metadata({}) == []

    def test_references_with_no_doi(self) -> None:
        metadata = {
            "dataset": {
                "description": {
                    "ReferencesAndLinks": [
                        "Smith et al., Nature Neuroscience, 2020"
                    ]
                }
            }
        }
        result = extract_dois_from_openneuro_metadata(metadata)
        assert result == []


class TestDoisToPaperIds:
    def test_basic_conversion(self) -> None:
        result = dois_to_paper_ids(["10.1038/s41593-019-0409-2"])
        assert result == ["paper:doi:10.1038:s41593-019-0409-2"]

    def test_multiple_dois(self) -> None:
        dois = ["10.1038/abc", "10.9999/xyz"]
        result = dois_to_paper_ids(dois)
        assert result == ["paper:doi:10.1038:abc", "paper:doi:10.9999:xyz"]

    def test_lowercase_applied(self) -> None:
        result = dois_to_paper_ids(["10.7554/eLife.60606"])
        assert result == ["paper:doi:10.7554:elife.60606"]

    def test_empty_list(self) -> None:
        assert dois_to_paper_ids([]) == []

    def test_slash_replaced_by_colon(self) -> None:
        result = dois_to_paper_ids(["10.1234/a/b/c"])
        assert result == ["paper:doi:10.1234:a:b:c"]
