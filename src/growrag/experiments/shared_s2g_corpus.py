"""Disk-backed, gold-free shared-paragraph retrieval and post-execution scoring.

中文：SQLite 保存倒排和原文，每次查询仅保留每文档一个 double 分数。不把全库
Counter/全文多次放进内存，不把同名但不同版本的段落合并。Gold 由独立入口后载入；
runtime 不保留 manifest、题型或文档归属。支持召回按原文版本和句子位置计算。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .hotpot import _answer_f1, _normalize
from .lexical_retriever import _finite_number, _tokens
from .protocol import RuntimeQuestion
from .s2g_author_api import AuthorDocument

INDEX_VERSION = "growrag-shared-paragraph-sqlite-bm25-v1"
MANIFEST_SCHEMAS = {
    "growrag-shared-hotpot-development-v1": "corpus_sources.jsonl",
    "growrag-shared-hotpot-scale-development-v1": "corpus_source_edges.jsonl",
}
RUNTIME_FILES = {"runtime_questions.jsonl", "corpus.jsonl"}
ARTIFACT_FILES = RUNTIME_FILES | {"gold.jsonl", "corpus_sources.jsonl", "corpus_source_edges.jsonl"}
_SHA = re.compile(r"[0-9a-f]{64}")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique)


def _ids_digest(values):
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _safe_file(root, name):
    if name not in ARTIFACT_FILES:
        raise ValueError("artifact is not allowlisted")
    path = (root / name).resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError("artifact escapes manifest directory")
    return path


def _manifest(path, expected_sha=None):
    path = Path(path).resolve(strict=True)
    if path.name != "manifest.json":
        raise ValueError("expected manifest.json with its hash sidecar")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    if sidecar != f"{digest}  manifest.json" or (
        expected_sha is not None and digest != expected_sha
    ):
        raise ValueError("manifest SHA mismatch")
    value = _json(raw)
    schema = value.get("schema_version")
    selection_safe = (
        value.get("selection_uses_gold") is False
        and value.get("corpus_selection_uses_gold") is False
        if schema == "growrag-shared-hotpot-development-v1"
        else value.get("selection_uses_gold_or_difficulty") is False
    )
    if (
        value.get("schema_version") not in MANIFEST_SCHEMAS
        or value.get("official_split") != "train"
        or value.get("role") != "development"
        or value.get("official_dev_test_used") is not False
        or not selection_safe
        or value.get("old_check_question_or_gold_projected") is not False
        or set(value.get("runtime_input_allowlist", [])) != RUNTIME_FILES
        or len(value.get("runtime_input_allowlist", [])) != 2
        or set(value.get("artifacts", {}))
        != RUNTIME_FILES | {"gold.jsonl", MANIFEST_SCHEMAS.get(schema)}
    ):
        raise ValueError("unreviewed manifest or runtime allowlist")
    ids = value.get("question_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(q, str) or not q.strip() for q in ids)
        or len(ids) != len(set(ids))
        or _ids_digest(ids) != value.get("selected_ids_sha256")
        or set(ids) != set(value.get("target_ids_forbidden_for_training_or_memory", []))
        or set(ids) & set(value.get("excluded_question_ids", []))
    ):
        raise ValueError("invalid/reserved-overlapping target allowlist")
    for name, artifact in value["artifacts"].items():
        if (
            not isinstance(artifact, dict)
            or artifact.get("runtime_safe") is not (name in RUNTIME_FILES)
            or artifact.get("contains_gold") is not (name == "gold.jsonl")
            or not isinstance(artifact.get("sha256"), str)
            or not _SHA.fullmatch(artifact["sha256"])
            or type(artifact.get("rows")) is not int
            or artifact["rows"] <= 0
            or type(artifact.get("bytes")) is not int
            or artifact["bytes"] <= 0
        ):
            raise ValueError("invalid artifact safety contract")
    return path.parent, value, digest


def _verify_file(root, manifest, name):
    path = _safe_file(root, name)
    info = manifest["artifacts"][name]
    if path.stat().st_size != info["bytes"] or _sha(path) != info["sha256"]:
        raise ValueError(f"artifact SHA/size mismatch: {name}")
    return path


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    doc_id: str
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self):
        return "\n".join(self.sentences)

    def as_author_document(self):
        return AuthorDocument(self.doc_id, self.title, self.text)


def _document(row):
    if not isinstance(row, dict) or set(row) != {"doc_id", "title", "sentences"}:
        raise ValueError("corpus may contain only doc_id/title/sentences")
    title, sentences = row["title"], row["sentences"]
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(sentences, list)
        or not sentences
        or not all(isinstance(s, str) for s in sentences)
        or not any(s.strip() for s in sentences)
        or row["doc_id"] != hashlib.sha256(_canonical([title, sentences])).hexdigest()
    ):
        raise ValueError("invalid document or exact-version document ID mismatch")
    return CorpusDocument(row["doc_id"], title, tuple(sentences))


def _configure(connection):
    connection.execute("PRAGMA cache_size=-4096")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")


class SharedBM25Index:
    """One persistent shared index; callback returns complete source paragraphs.

    The same positive-IDF formula, tokenization and doc-ID tie break as the existing
    BM25SentenceRetriever, with each indexing unit now a complete paragraph.
    """

    def __init__(self, path, corpus_path, corpus_sha256, document_count, *, k1=1.2, b=0.75):
        self.k1, self.b = _finite_number(k1, "k1"), _finite_number(b, "b")
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.path = Path(path).resolve()
        self.corpus_sha256 = corpus_sha256
        self.document_count = document_count
        self.retrieval_config = {
            "algorithm": INDEX_VERSION,
            "unit": "complete_paragraph",
            "k1": self.k1,
            "b": self.b,
            "tokenization": "unicode_alphanumeric_casefold_no_stemming",
            "query_terms": "distinct_sorted",
            "idf": "log1p_positive_robertson",
            "zero_score": "exclude",
            "tie_break": "doc_id",
        }
        self.context_fingerprint = hashlib.sha256(
            _canonical(
                {
                    "corpus_sha256": corpus_sha256,
                    **self.retrieval_config,
                }
            )
        ).hexdigest()
        if self.path == Path(corpus_path).resolve():
            raise ValueError("index path cannot overwrite corpus")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._build(Path(corpus_path))
        # 已存在的不完整/异源缓存只拒绝，不偷偷删除或覆盖。
        if (
            _sha(self.path)
            != self.path.with_suffix(self.path.suffix + ".sha256").read_text().strip()
        ):
            raise ValueError("index SHA mismatch or incomplete index")
        self._db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        _configure(self._db)
        meta = dict(self._db.execute("SELECT key,value FROM meta"))
        if (
            meta.get("ready") != "1"
            or meta.get("version") != INDEX_VERSION
            or meta.get("corpus_sha256") != corpus_sha256
            or meta.get("context_fingerprint") != self.context_fingerprint
            or int(meta.get("document_count", -1)) != document_count
        ):
            self._db.close()
            raise ValueError("index incomplete or source/config mismatch")
        self.average_length = float(meta["average_length"])

    def _build(self, corpus_path):
        # Exclusive placeholder preserves partial builds for diagnosis after failure.
        self.path.touch(exist_ok=False)
        db = sqlite3.connect(self.path)
        _configure(db)
        try:
            db.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE docs(n INTEGER PRIMARY KEY,doc_id TEXT UNIQUE NOT NULL,
                                  title TEXT NOT NULL,sentences TEXT NOT NULL,
                                  length INTEGER NOT NULL);
                CREATE TABLE postings(term TEXT NOT NULL,n INTEGER NOT NULL,tf INTEGER NOT NULL,
                                      PRIMARY KEY(term,n)) WITHOUT ROWID;
                CREATE TABLE terms(term TEXT PRIMARY KEY,df INTEGER NOT NULL) WITHOUT ROWID;
            """)
            total_length = 0
            count = 0
            with corpus_path.open("rb") as handle:
                for count, raw in enumerate(handle, start=1):
                    doc = _document(_json(raw))
                    counts = Counter(_tokens(f"{doc.title} {doc.text}"))
                    length = sum(counts.values())
                    total_length += length
                    db.execute(
                        "INSERT INTO docs VALUES(?,?,?,?,?)",
                        (
                            count,
                            doc.doc_id,
                            doc.title,
                            json.dumps(doc.sentences, ensure_ascii=False),
                            length,
                        ),
                    )
                    db.executemany(
                        "INSERT INTO postings VALUES(?,?,?)",
                        ((term, count, tf) for term, tf in counts.items()),
                    )
                    if count % 1000 == 0:
                        db.commit()
            if (
                count != self.document_count
                or not total_length
                or _sha(corpus_path) != self.corpus_sha256
            ):
                raise ValueError(
                    "corpus changed or row count/token contract failed during indexing"
                )
            db.execute("INSERT INTO terms SELECT term,COUNT(*) FROM postings GROUP BY term")
            metadata = {
                "ready": "1",
                "version": INDEX_VERSION,
                "corpus_sha256": self.corpus_sha256,
                "context_fingerprint": self.context_fingerprint,
                "document_count": str(count),
                "average_length": str(total_length / count),
            }
            db.executemany("INSERT INTO meta VALUES(?,?)", metadata.items())
            db.commit()
        finally:
            db.close()
        with self.path.with_suffix(self.path.suffix + ".sha256").open(
            "x", encoding="ascii"
        ) as handle:
            handle.write(_sha(self.path) + "\n")

    def document(self, doc_id):
        row = self._db.execute(
            "SELECT title,sentences FROM docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return CorpusDocument(doc_id, row[0], tuple(_json(row[1])))

    def search_with_scores(self, query, k):
        if not isinstance(query, str):
            raise TypeError("query must be text")
        terms = tuple(sorted(set(_tokens(query))))
        if not terms:
            raise ValueError("query must contain alphanumeric tokens")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be positive integer")
        scores = array("d", [0.0]) * (self.document_count + 1)
        for term in terms:
            frequency = self._db.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
            if frequency is None:
                continue
            df = frequency[0]
            idf = math.log1p((self.document_count - df + 0.5) / (df + 0.5))
            cursor = self._db.execute(
                "SELECT p.n,p.tf,d.length FROM postings p JOIN docs d ON d.n=p.n WHERE p.term=?",
                (term,),
            )
            for number, tf, length in cursor:
                norm = self.k1 * (1 - self.b + self.b * length / self.average_length)
                scores[number] += idf * (tf * (self.k1 + 1) / (tf + norm))
        ranked = heapq.nsmallest(
            k,
            (
                (-scores[n], doc_id)
                for n, doc_id in self._db.execute("SELECT n,doc_id FROM docs")
                if scores[n] > 0
            ),
        )
        return tuple(
            (self.document(doc_id).as_author_document(), -score) for score, doc_id in ranked
        )

    def __call__(self, query, k):
        return tuple(doc for doc, _ in self.search_with_scores(query, k))

    def close(self):
        self._db.close()


@dataclass(frozen=True, slots=True)
class SharedRuntime:
    questions: tuple[RuntimeQuestion, ...]
    index: SharedBM25Index
    metadata: dict

    def close(self):
        self.index.close()


def load_runtime(manifest_path, *, index_path=None, k1=1.2, b=0.75, expected_manifest_sha256=None):
    """Load ONLY allowlisted question/corpus files; never read gold or owner artifacts."""
    root, manifest, digest = _manifest(manifest_path, expected_manifest_sha256)
    questions_path = _verify_file(root, manifest, "runtime_questions.jsonl")
    corpus_path = _verify_file(root, manifest, "corpus.jsonl")
    questions = []
    with questions_path.open("rb") as handle:
        for raw in handle:
            row = _json(raw)
            if not isinstance(row, dict) or set(row) != {"question_id", "text", "dataset"}:
                raise ValueError("runtime question includes unallowlisted fields")
            questions.append(RuntimeQuestion(**row))
    if [q.question_id for q in questions] != manifest["question_ids"] or len(questions) != manifest[
        "artifacts"
    ]["runtime_questions.jsonl"]["rows"]:
        raise ValueError("runtime question order/allowlist/row count mismatch")
    count = manifest["artifacts"]["corpus.jsonl"]["rows"]
    if count != manifest["corpus_statistics"]["documents"]:
        raise ValueError("corpus manifest counts disagree")
    chosen_path = Path(index_path) if index_path else root / "bm25_shared.sqlite"
    if chosen_path.resolve() in {
        root / name for name in ARTIFACT_FILES | {"manifest.json", "manifest.sha256"}
    }:
        raise ValueError("index path overlaps source artifacts")
    index = SharedBM25Index(
        chosen_path, corpus_path, manifest["artifacts"]["corpus.jsonl"]["sha256"], count, k1=k1, b=b
    )
    return SharedRuntime(
        tuple(questions),
        index,
        {
            "manifest_sha256": digest,
            "corpus_sha256": index.corpus_sha256,
            "document_count": count,
            "question_count": len(questions),
            "index_path": str(index.path),
            "retrieval_config": dict(index.retrieval_config),
            "context_fingerprint": index.context_fingerprint,
        },
    )


@dataclass(frozen=True, slots=True)
class ExactSupport:
    doc_id: str
    title: str
    sentence_index: int
    text_sha256: str


@dataclass(frozen=True, slots=True)
class ExactGold:
    question_id: str
    answers: tuple[str, ...]
    exact_support: tuple[ExactSupport, ...]
    annotation_status: str = "valid"
    annotation_issue: str | None = None


def load_gold_after_execution(
    manifest_path, *, completed_question_ids, expected_manifest_sha256=None
):
    """Separate scoring authority; callers attest both arms finished before using it.

    This call is not a cryptographic proof of execution. It exposes only the explicitly
    completed IDs, never stores gold on SharedRuntime or its retriever.
    """
    root, manifest, _ = _manifest(manifest_path, expected_manifest_sha256)
    completed = tuple(completed_question_ids)
    if (
        not completed
        or len(completed) != len(set(completed))
        or not set(completed) <= set(manifest["question_ids"])
    ):
        raise ValueError("explicit unique completed question allowlist required")
    path = _verify_file(root, manifest, "gold.jsonl")
    result, seen = {}, []
    with path.open("rb") as handle:
        for raw in handle:
            row = _json(raw)
            allowed = {
                "question_id",
                "answers",
                "supporting_facts",
                "exact_support",
                "annotation_status",
                "annotation_issue",
            }
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError("unexpected gold fields")
            qid = row.get("question_id")
            seen.append(qid)
            if qid not in completed:
                continue
            status = row.get("annotation_status", "valid")
            if status not in {"valid", "invalid"}:
                raise ValueError("invalid annotation status")
            answers = row.get("answers", [])
            if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
                raise ValueError("invalid gold answers")
            support = []
            if status == "valid":
                if not answers or not row.get("exact_support"):
                    raise ValueError("valid annotation lacks answer/support")
                for item in row["exact_support"]:
                    if (
                        not isinstance(item, dict)
                        or set(item) != {"doc_id", "title", "sentence_index", "text_sha256"}
                        or not _SHA.fullmatch(item["doc_id"])
                        or not _SHA.fullmatch(item["text_sha256"])
                        or type(item["sentence_index"]) is not int
                        or item["sentence_index"] < 0
                        or not isinstance(item["title"], str)
                    ):
                        raise ValueError("invalid exact support pointer")
                    support.append(ExactSupport(**item))
            result[qid] = ExactGold(
                qid, tuple(answers), tuple(support), status, row.get("annotation_issue")
            )
    if seen != manifest["question_ids"] or len(seen) != manifest["artifacts"]["gold.jsonl"]["rows"]:
        raise ValueError("gold question order/allowlist/row count mismatch")
    return result


def _author_units_with_spans(doc):
    """Mirror ONLY regex-fallback segmentation for provenance alignment, not retrieval.

    Exact parity against pinned split_wiki_sentences is tested. No learned judgment is
    made: returned spans simply locate the original characters selected by the author.
    """
    text = doc.text.replace("\r\n", "\n").replace("\r", "\n")
    units, cursor = [], 0
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        cjk = len(re.findall(r"[\u4e00-\u9fff]", paragraph))
        pattern = (
            r"(?<=[。！？!?])"
            if cjk and cjk / max(1, len(paragraph)) > 0.02
            else r"(?<=[.!?])\s+|[\n]+"
        )
        for part in re.split(pattern, paragraph):
            part = part.strip()
            if part:
                start = text.find(part, cursor)
                if start < 0:
                    raise ValueError("cannot locate author sentence in exact document")
                units.append((part, start, start + len(part)))
                cursor = start + len(part)
    return text, units


def _covered_raw_sentences(doc, selected):
    text, units = _author_units_with_spans(doc)
    spans, unaligned = [], 0
    for source in selected:
        sid = source.get("sentence_id")
        if (
            source.get("title") != doc.title
            or type(sid) is not int
            or not 1 <= sid <= len(units)
            or source.get("text") != units[sid - 1][0]
        ):
            unaligned += 1
        else:
            spans.append(units[sid - 1][1:])
    spans.sort()
    covered, partial, offset = set(), 0, 0
    for number, raw in enumerate(doc.sentences):
        sentence = raw.replace("\r\n", "\n").replace("\r", "\n")
        start, end = offset, offset + len(sentence)
        offset = end + 1
        if not sentence.strip():
            continue
        intersections = [(max(start, a), min(end, b)) for a, b in spans if a < end and b > start]
        position = start
        missing = False
        for a, b in intersections:
            if text[position:a].strip():
                missing = True
            position = max(position, b)
        missing |= bool(text[position:end].strip())
        if not missing:
            covered.add((doc.doc_id, number))
        elif intersections:
            partial += 1
    return covered, unaligned, partial


def score_result(result, gold: ExactGold, index: SharedBM25Index, *, retained_mode):
    """Version-exact annotation coverage AFTER execution, not semantic entailment."""
    if retained_mode not in {"sources", "raw"}:
        raise ValueError("retained_mode must explicitly be sources or raw")
    if result.get("question_id") != gold.question_id:
        raise ValueError("result/gold question mismatch")
    if gold.annotation_status != "valid":
        return {
            "question_id": gold.question_id,
            "unscorable_annotation": True,
            "annotation_status": gold.annotation_status,
            "annotation_issue": gold.annotation_issue,
            "answer_em": None,
            "answer_f1": None,
            "raw_support_recall": None,
            "retained_support_recall": None,
            "unaligned_sources": 0,
        }
    targets = set()
    for support in gold.exact_support:
        doc = index.document(support.doc_id)
        if (
            doc.title != support.title
            or support.sentence_index >= len(doc.sentences)
            or hashlib.sha256(doc.sentences[support.sentence_index].encode()).hexdigest()
            != support.text_sha256
        ):
            raise ValueError("gold exact document/sentence fingerprint mismatch")
        targets.add((support.doc_id, support.sentence_index))
    raw_refs, valid_docs, bad_docs = set(), {}, 0
    for row in result.get("retrieved_documents", []):
        try:
            doc = index.document(row.get("doc_id"))
        except (KeyError, TypeError):
            bad_docs += 1
            continue
        if row.get("title") != doc.title or row.get("text") != doc.text:
            bad_docs += 1
            continue
        valid_docs[doc.doc_id] = doc
        raw_refs.update(
            (doc.doc_id, i) for i, sentence in enumerate(doc.sentences) if sentence.strip()
        )
    unaligned, partial = 0, 0
    if retained_mode == "raw":
        retained = raw_refs
    else:
        retained, by_doc = set(), {}
        for source in result.get("sources", []):
            doc_id = source.get("doc_id")
            if doc_id not in valid_docs:
                unaligned += 1
            else:
                by_doc.setdefault(doc_id, []).append(source)
        for doc_id, selected in by_doc.items():
            covered, bad, incomplete = _covered_raw_sentences(valid_docs[doc_id], selected)
            retained.update(covered)
            unaligned += bad
            partial += incomplete
    answer = result.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("answer must be text")
    return {
        "question_id": gold.question_id,
        "unscorable_annotation": False,
        "annotation_status": "valid",
        "answer_em": float(any(_normalize(answer) == _normalize(a) for a in gold.answers)),
        "answer_f1": max(_answer_f1(answer, a) for a in gold.answers),
        "raw_support_recall": len(raw_refs & targets) / len(targets) if targets else None,
        "retained_support_recall": len(retained & targets) / len(targets) if targets else None,
        "raw_support_hits": len(raw_refs & targets),
        "retained_support_hits": len(retained & targets),
        "gold_support_count": len(targets),
        "unaligned_sources": unaligned,
        "partial_raw_sentences": partial,
        "unaligned_retrieved_documents": bad_docs,
        "retained_mode": retained_mode,
        "answer_supported": None,
        "coverage_notice": "Exact annotated evidence coverage is not a semantic support judgment.",
    }
