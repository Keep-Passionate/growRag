"""Build the explicitly curated, offline Zotero reading pack; no model calls.

Metadata snapshots and annotations are reviewed inputs, not generated citations.
Only files beside this script are written. Existing reading packages are untouched.
"""

from __future__ import annotations

import hashlib
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
    "z": "http://www.zotero.org/namespaces/export#",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

GROUPS = {
    "A": "方向一｜经验辅助的低成本查询选择",
    "B": "方向二｜保留适用边界的紧凑经验",
    "C": "方向三｜值得长期持有的稳定记忆",
    "D": "方向四｜证据驱动的有限修复循环",
}
FIRST = [
    "qpp_variant",
    "reformir",
    "raqg_qpp",
    "reformer",
    "rrm",
    "qcr",
    "useful_memories",
    "continual_memory",
    "erm",
    "sufficient_context",
    "s2g_rag",
]
EXTRA_URLS = {
    "qpp_variant": "https://arxiv.org/html/2604.22661v1",
    "reformer": "https://arxiv.org/html/2604.01417v1",
    "reformir": "https://arxiv.org/html/2605.00560v1",
    "raqg_qpp": "https://eprints.gla.ac.uk/385540/",
    "wewrite": "https://arxiv.org/html/2602.17667v2",
    "right_track": "https://arxiv.org/html/2507.10411v1",
    "rag_qpp_tois": "https://github.com/APARAJITA1997/RAG_QPP_JH_2026",
    "useful_memories": "https://arxiv.org/html/2605.12978v2",
}


def q(name: str) -> str:
    prefix, local = name.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def sub(parent: ET.Element, name: str, value=None, **attributes) -> ET.Element:
    element = ET.SubElement(parent, q(name), {q(k): v for k, v in attributes.items()})
    if value is not None:
        element.text = str(value)
    return element


def uid(slug: str) -> str:
    return f"urn:growrag:reading:20260908:{slug}"


def guide(record: dict, note: dict, number: int) -> str:
    content = [f"<div><h2>{number:02d}｜{html.escape(note['short'])}</h2>"]
    sections = [
        ("阅读级别与方向", note["priority"] + "；" + " / ".join(GROUPS[g] for g in note["groups"])),
        ("正式发表状态", note["venue_label"]),
        ("做了什么；要学什么", note["learn"]),
        ("没有据此证明什么；与我们的重合边界", note["boundary"]),
        ("精读位置", note["read"]),
        ("读完应能回答", note["question"]),
    ]
    for heading, body in sections:
        content.append(f"<h3>{html.escape(heading)}</h3><p>{html.escape(body)}</p>")
    content.append(
        "<p>以上是 AI 辅助导读，不是论文原摘要，不代表用户已经读过。"
        "候选创新尚未证明。未计量引用次数，不作虚构高引排名。</p>"
    )
    for label, link in [
        ("论文入口", record["url"]),
        ("元数据核验来源", record["verification_source"]),
        ("补充原文或作者资料", EXTRA_URLS.get(record["slug"])),
    ]:
        if link:
            content.append(f'<p><a href="{html.escape(link, quote=True)}">{label}</a></p>')
    content.append(
        f"<p>元数据核验日期：{record['verification_date']}；导读整理：2026-09-08。历史日期原样保留，不表示全部论文今天逐句重读。</p>"
    )
    if record.get("verification_note"):
        content.append(f"<p>{html.escape(record['verification_note'])}</p>")
    content.append("<p>仅确认预印本不等于未投稿或已被拒；不含 PDF，不直接写入 Zotero。</p></div>")
    return "".join(content)


def emit(records: list[dict], notes: dict, name: str) -> dict:
    root = ET.Element(q("rdf:RDF"))
    for number, record in enumerate(records, 1):
        slug = record["slug"]
        note = notes[slug]
        item = sub(root, "rdf:Description", **{"rdf:about": uid(slug)})
        sub(item, "z:itemType", record["type"])
        sub(item, "dc:title", record["title"])
        sub(item, "dc:date", record["year"])
        authors = sub(sub(item, "bib:authors"), "rdf:Seq")
        for author in record["authors"]:
            person = sub(sub(authors, "rdf:li"), "foaf:Person")
            sub(person, "foaf:givenName", author["given"])
            sub(person, "foaf:surname", author["family"])
        uri = sub(sub(item, "dc:identifier"), "dcterms:URI")
        sub(uri, "rdf:value", record["url"])
        host = sub(sub(item, "dcterms:isPartOf"), "bib:Journal")
        sub(host, "dc:title", record["venue"])
        if record.get("doi"):
            sub(host, "dc:identifier", "DOI " + record["doi"])
        if record.get("volume"):
            sub(host, "prism:volume", record["volume"])
        if record.get("issue"):
            sub(host, "prism:number", record["issue"])
        if record.get("pages"):
            sub(item, "bib:pages", record["pages"])
        sub(
            item,
            "dc:description",
            note["venue_label"]
            + "; AI 中文导读见子笔记。元数据核验日期 "
            + record["verification_date"],
        )
        sub(item, "dc:subject", "GrowRAG/研究方向2026-09-08")
        sub(item, "dc:subject", "阅读/" + note["priority"])
        for group in note["groups"]:
            sub(item, "dc:subject", GROUPS[group])
        if slug in FIRST:
            sub(item, "dc:subject", f"本轮顺序/{FIRST.index(slug) + 1:02d}")
        sub(item, "dcterms:isReferencedBy", **{"rdf:resource": uid(slug + ":guide")})
        memo = sub(root, "bib:Memo", **{"rdf:about": uid(slug + ":guide")})
        sub(memo, "rdf:value", guide(record, note, number))
    overview = sub(root, "bib:Memo", **{"rdf:about": uid(name + ":overview")})
    sub(
        overview,
        "rdf:value",
        "<div><h2>00｜阅读导航</h2><p>这是研究选题包，不要求逐篇全部精读。"
        "方向一、二优先；三是可独立研究的备选；四先作支撑模块。"
        "先读 QPP → ReformIR → RAQG-QPP → ReFormeR → RRM → QCR，"
        "再决定是否转向长期记忆。</p><p>有基础再按此顺序读；"
        "术语不熟先补 Query Rewriting、ReAct、Self-RAG。"
        "完整包与新增包二选一导入；不同组中的同一文献共享条目。"
        "不要覆盖或删除原有批注。</p><p>我们要检验的是历史相对现场修复的额外价值，"
        "不把少调用、永不用记忆或更高预测分数自动称作成功。"
        "两篇 TOIS 已确认正式期刊；Findings/Industry/Workshop 分开标注。</p></div>",
    )
    collection = sub(root, "z:Collection", **{"rdf:about": uid(name + ":collection")})
    sub(collection, "dc:title", f"GrowRAG｜{name}｜2026-09-08")
    sub(collection, "dcterms:hasPart", **{"rdf:resource": uid(name + ":overview")})
    for group, title in GROUPS.items():
        members = [r for r in records if group in notes[r["slug"]]["groups"]]
        if not members:
            continue
        group_id = uid(name + ":group:" + group)
        sub(collection, "dcterms:hasPart", **{"rdf:resource": group_id})
        group_node = sub(root, "z:Collection", **{"rdf:about": group_id})
        sub(group_node, "dc:title", title)
        for record in members:
            sub(group_node, "dcterms:hasPart", **{"rdf:resource": uid(record["slug"])})
    ET.indent(root, space="  ")
    path = ROOT / f"GrowRAG_{name}_2026-09-08.rdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return validate(path, len(records))


def validate(path: Path, expected: int) -> dict:
    root = ET.parse(path).getroot()
    ids = [e.attrib[q("rdf:about")] for e in root if q("rdf:about") in e.attrib]
    assert len(ids) == len(set(ids)), "duplicate IDs"
    references = [e.attrib[q("rdf:resource")] for e in root.iter() if q("rdf:resource") in e.attrib]
    assert all(ref in ids for ref in references), "dangling RDF relation"
    items = [e for e in root if e.find(q("z:itemType")) is not None]
    assert len(items) == expected
    titles = [e.findtext(q("dc:title")) for e in items]
    assert len(set(titles)) == expected
    for item in items:
        assert item.find("bib:authors/rdf:Seq/rdf:li/foaf:Person", NS) is not None
        assert item.findtext("dc:identifier/dcterms:URI/rdf:value", namespaces=NS)
        assert len(item.findall(q("dcterms:isReferencedBy"))) == 1
    memos = root.findall(q("bib:Memo"))
    assert len(memos) == expected + 1
    for memo in memos:
        ET.fromstring(memo.findtext(q("rdf:value")))
    return {
        "file": path.name,
        "items": expected,
        "notes": len(memos),
        "collections": len(root.findall(q("z:Collection"))),
        "types": dict(Counter(e.findtext(q("z:itemType")) for e in items)),
        "dangling_relations": 0,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "zotero_client_import_tested": False,
        "pdf_attachments": 0,
    }


def main() -> None:
    base = json.loads((ROOT / "metadata_base.json").read_text(encoding="utf-8"))
    new = json.loads((ROOT / "metadata_new.json").read_text(encoding="utf-8"))
    notes = json.loads((ROOT / "reading_notes.json").read_text(encoding="utf-8"))
    records = base + new
    assert len(records) == 30
    assert len({r["slug"] for r in records}) == 30
    assert set(notes) == {r["slug"] for r in records}
    order = FIRST + [r["slug"] for r in records if r["slug"] not in FIRST]
    lookup = {r["slug"]: r for r in records}
    records = [lookup[s] for s in order]
    reports = [
        emit(records, notes, "四个研究方向_精选30篇"),
        emit([r for r in records if r["slug"] in {n["slug"] for n in new}], notes, "本轮新增8篇"),
    ]
    (ROOT / "validation.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 逐篇导读：四个研究方向",
        "",
        "日期：2026-09-08。AI 辅助整理；正文是阅读任务，不是全部方法均已实现或全文逐句核验的声明。",
        "",
        "元数据来自旧精选包和本轮出版社/官方登记；验证日期逐条保留。以论文原文为准。",
        "",
    ]
    for group, title in GROUPS.items():
        lines += [f"## {title}", "", "| 阅读级别 | 论文 | 学习重点 |", "|---|---|---|"]
        for record in records:
            note = notes[record["slug"]]
            if group in note["groups"]:
                lines.append(
                    f"| {note['priority']} | [{note['short']}]({record['url']}) | {note['learn']} |"
                )
        lines.append("")
    for number, record in enumerate(records, 1):
        note = notes[record["slug"]]
        lines += [
            f"## {number:02d}. {note['short']}",
            "",
            f"[{record['title']}]({record['url']})",
            "",
            f"- 级别：{note['priority']}；{note['venue_label']}。",
            f"- 做了什么、学什么：{note['learn']}",
            f"- 重合与边界：{note['boundary']}",
            f"- 精读：{note['read']}。",
            f"- 自检：{note['question']}",
            f"- 元数据核验：{record['verification_date']}；"
            f"[来源]({record['verification_source']})。",
            "",
        ]
        if record["slug"] in EXTRA_URLS:
            lines += [f"[补充原文/作者资料]({EXTRA_URLS[record['slug']]})", ""]
    (ROOT / "逐篇导读.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
