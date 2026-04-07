"""PubMed E-utilities client for search and abstract retrieval."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from parkinson_ai.data.base_client import AsyncAPIClient


class PubMedArticle(BaseModel):
    """Structured PubMed article record."""

    pmid: str
    title: str
    abstract: str = ""
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    mesh_terms: list[str] = Field(default_factory=list)


class PubMedClient(AsyncAPIClient):
    """Client for NCBI PubMed E-utilities."""

    def __init__(self) -> None:
        super().__init__("https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=2.5)

    async def search(
        self,
        query: str,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> list[str]:
        """Search PubMed and return PMIDs."""

        xml_text = await self.get_text(
            "/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "xml",
                "retmax": retmax,
                "sort": sort,
            },
        )
        root = ET.fromstring(xml_text)
        return [element.text or "" for element in root.findall(".//Id") if element.text]

    async def fetch_articles(self, pmids: Sequence[str]) -> list[PubMedArticle]:
        """Fetch article details for a list of PMIDs."""

        if not pmids:
            return []
        xml_text = await self.get_text(
            "/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        return list(self.parse_efetch_xml(xml_text))

    def parse_efetch_xml(self, xml_text: str) -> Iterable[PubMedArticle]:
        """Parse PubMed efetch XML into article models."""

        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid = _find_text(article, ".//PMID")
            title = _find_text(article, ".//ArticleTitle")
            abstract = " ".join(str(part.text).strip() for part in article.iterfind(".//AbstractText") if part.text and part.text.strip())
            journal = _find_text(article, ".//Journal/Title")
            year_text = _find_text(article, ".//PubDate/Year")
            authors = []
            for author in article.findall(".//Author"):
                last_name = _find_text(author, "./LastName")
                fore_name = _find_text(author, "./ForeName")
                if last_name or fore_name:
                    authors.append(" ".join(item for item in [fore_name, last_name] if item))
            mesh_terms = [descriptor.text or "" for descriptor in article.findall(".//MeshHeading/DescriptorName") if descriptor.text]
            yield PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal or None,
                authors=authors,
                publication_year=int(year_text) if year_text.isdigit() else None,
                mesh_terms=mesh_terms,
            )


def _find_text(element: ET.Element, path: str) -> str:
    """Find nested text from an XML element."""

    found = element.find(path)
    return found.text.strip() if found is not None and found.text else ""
