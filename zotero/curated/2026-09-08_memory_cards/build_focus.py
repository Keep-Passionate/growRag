"""Offline regrouping of verified bibliography records, not runtime RAG code."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "2026-09-08_directions"
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "z": "http://www.zotero.org/namespaces/export#",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}
GROUPS = {
    "representation": "01 核心问题｜经验表示与适用性（含当前状态基础）",
    "selection": "02 配套问题｜经验辅助的低成本选择",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def q(name: str) -> str:
    prefix, local = name.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def uid(slug: str) -> str:
    return f"urn:growrag:memory-cards:20260908:{slug}"


def add(parent: ET.Element, name: str, value=None, **attributes) -> ET.Element:
    node = ET.SubElement(parent, q(name), {q(k): v for k, v in attributes.items()})
    if value is not None:
        node.text = str(value)
    return node


def main() -> None:
    plan = json.loads((ROOT / "reading_plan.json").read_text(encoding="utf-8"))
    metadata = []
    for filename in ("metadata_base.json", "metadata_new.json"):
        metadata.extend(json.loads((PREVIOUS / filename).read_text(encoding="utf-8")))
    lookup = {record["slug"]: record for record in metadata}
    # Author order, DOI and pages checked against the official AAAI record.
    lookup["expel"] = {
        "slug": "expel",
        "title": "ExpeL: LLM Agents Are Experiential Learners",
        "year": 2024,
        "type": "conferencePaper",
        "authors": [
            {"given": "Andrew", "family": "Zhao"},
            {"given": "Daniel", "family": "Huang"},
            {"given": "Quentin", "family": "Xu"},
            {"given": "Matthieu", "family": "Lin"},
            {"given": "Yong-Jin", "family": "Liu"},
            {"given": "Gao", "family": "Huang"},
        ],
        "venue": "Proceedings of the AAAI Conference on Artificial Intelligence",
        "volume": "38",
        "issue": "17",
        "pages": "19632–19642",
        "doi": "10.1609/aaai.v38i17.29936",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/29936",
        "verification_source": "https://ojs.aaai.org/index.php/AAAI/article/view/29936",
        "verification_date": "2026-09-08",
        "source_rdf": "zotero/current/GrowRAG_核心精读包_合并更新_2026-08-14.rdf",
    }
    records = [copy.deepcopy(lookup[slug]) for slug in plan]
    assert len(records) == 12
    tree = ET.Element(q("rdf:RDF"))
    for record in records:
        slug = record["slug"]
        note = plan[slug]
        item = add(tree, "rdf:Description", **{"rdf:about": uid(slug)})
        add(item, "z:itemType", record["type"])
        add(item, "dc:title", record["title"])
        add(item, "dc:date", record["year"])
        people = add(add(item, "bib:authors"), "rdf:Seq")
        for author in record["authors"]:
            person = add(add(people, "rdf:li"), "foaf:Person")
            add(person, "foaf:givenName", author["given"])
            add(person, "foaf:surname", author["family"])
        uri = add(add(item, "dc:identifier"), "dcterms:URI")
        add(uri, "rdf:value", record["url"])
        host = add(add(item, "dcterms:isPartOf"), "bib:Journal")
        add(host, "dc:title", record["venue"])
        if record.get("doi"):
            add(host, "dc:identifier", "DOI " + record["doi"])
        for field, tag in (("volume", "prism:volume"), ("issue", "prism:number")):
            if record.get(field):
                add(host, tag, record[field])
        if record.get("pages"):
            add(item, "bib:pages", record["pages"])
        add(item, "dc:description", note["role"])
        add(item, "dc:subject", "GrowRAG/两条主线与经验表示")
        add(item, "dc:subject", GROUPS[note["group"]])
        order = f"精读顺序/{note['order']:02d}" if note["order"] else "按需插读"
        add(item, "dc:subject", order)
        add(item, "dcterms:isReferencedBy", **{"rdf:resource": uid(slug + ":note")})
        memo = add(tree, "bib:Memo", **{"rdf:about": uid(slug + ":note")})
        parts = [
            (order, note["role"]),
            ("怎么读", note["focus"]),
            ("已有覆盖与边界", note["boundary"]),
        ]
        text = "<div>" + "".join(
            f"<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p>" for title, body in parts
        )
        text += f'<p><a href="{html.escape(note["method_url"], quote=True)}">方法阅读入口</a></p>'
        text += (
            "<p>这是 AI 辅助导读与选题建议，不是论文原摘要或用户已读声明。"
            "元数据沿用日期：" + record["verification_date"] + "；本次导读：2026-09-08。"
            "方法链接可能为作者公开版本，不表示逐字核对最终正式 PDF。"
            "预印本不等于未投稿；未统计引用次数；无 PDF 附件。</p></div>"
        )
        add(memo, "rdf:value", text)
    overview = add(tree, "bib:Memo", **{"rdf:about": uid("overview")})
    add(
        overview,
        "rdf:value",
        "<div><h2>两条主线与新的阅读顺序</h2>"
        "<p>核心：经验怎样记才能判断适用；配套：怎样借助经验低成本选择。"
        "有限修复循环是使用场景，不另列第三研究方向；长期更新后置。</p>"
        "<p>先读 ReFormeR → ReMe → ExpeL，再 S2G → RRM → QCR，"
        "最后 QPP → ReformIR → RAQG-QPP。AWM、GAM、Sufficient Context 按需插读。</p>"
        "<p>12篇是旧推荐重组，10篇已审稿（含ECIR与Findings）、2篇预印本，"
        "不是12篇新发现。已导入旧包可仅按新导读阅读，不需要重复导入。"
        "不要删除旧条目或批注。RDF结构验证不等于Zotero界面导入测试。</p></div>",
    )
    parent = add(tree, "z:Collection", **{"rdf:about": uid("collection")})
    add(parent, "dc:title", "GrowRAG｜两条主线聚焦12篇｜2026-09-08后续")
    add(parent, "dcterms:hasPart", **{"rdf:resource": uid("overview")})
    for group, title in GROUPS.items():
        add(parent, "dcterms:hasPart", **{"rdf:resource": uid(group)})
        collection = add(tree, "z:Collection", **{"rdf:about": uid(group)})
        add(collection, "dc:title", title)
        for slug, note in plan.items():
            if note["group"] == group:
                add(collection, "dcterms:hasPart", **{"rdf:resource": uid(slug)})
    ET.indent(tree, space="  ")
    output = ROOT / "GrowRAG_两条主线_经验表示与低成本选择_12篇.rdf"
    ET.ElementTree(tree).write(output, encoding="utf-8", xml_declaration=True)
    parsed = ET.parse(output).getroot()
    ids = [element.attrib[q("rdf:about")] for element in parsed]
    assert len(ids) == len(set(ids))
    assert all(
        element.attrib[q("rdf:resource")] in ids
        for element in parsed.iter()
        if q("rdf:resource") in element.attrib
    )
    items = [element for element in parsed if element.find(q("z:itemType")) is not None]
    assert len(items) == 12
    assert len(parsed.findall(q("z:Collection"))) == 3
    for item, record in zip(items, records, strict=True):
        assert item.findtext(q("dc:title")) == record["title"]
        assert item.findtext("dc:identifier/dcterms:URI/rdf:value", namespaces=NS) == record["url"]
        assert len(item.findall("bib:authors/rdf:Seq/rdf:li", NS)) == len(record["authors"])
        assert len(item.findall(q("dcterms:isReferencedBy"))) == 1
    memos = parsed.findall(q("bib:Memo"))
    assert len(memos) == 13
    for memo in memos:
        ET.fromstring(memo.findtext(q("rdf:value")))
    report = {
        "file": output.name,
        "papers": len(items),
        "notes": len(memos),
        "collections": 3,
        "types": dict(Counter(record["type"] for record in records)),
        "dangling_relations": 0,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "zotero_client_import_tested": False,
        "pdf_attachments": 0,
    }
    for filename, data in (("metadata_snapshot.json", records), ("validation.json", report)):
        (ROOT / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
