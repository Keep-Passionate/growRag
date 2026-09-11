"""Build offline reading assets only; no network, model call, or runtime RAG edits.

Adapted from ../2026-09-08_memory_cards/build_focus.py. The prior snapshot is
read-only; all output paths are in this script's own dated directory.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "2026-09-08_memory_cards" / "metadata_snapshot.json"
DATE = "2026-09-11"
FULL_NAME = "GrowRAG_最小表示选择实验_完整17篇_2026-09-11.rdf"
INCREMENT_NAME = "GrowRAG_相对09-08聚焦包_补充5篇_2026-09-11.rdf"
GROUPS = {
    "core": "01 最小实验前先读｜方法与数据4篇",
    "overlap": "02 创新排重与经验基础｜3篇",
    "selector": "03 后续选择器增强｜QPP相关3篇",
    "loop": "04 有限修复循环后置｜3篇",
    "structure": "05 结构分支按需读｜4篇",
}
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "z": "http://www.zotero.org/namespaces/export#",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
}
for prefix, namespace in NS.items():
    ET.register_namespace(prefix, namespace)

PROTOCOL = (
    "方向一是主任务：怎样选到值得执行的改写；方向二是配套：经验必须保留哪些适用信息。"
    "阅读集合是用途分类，不是新增研究方向。首轮做 E1（只改变选择器所读经验表示，"
    "同源候选、选择器、当前问题和被选动作的规范执行保持一致）与 E3（REUSE−FRESH 归因）。"
    "E2 是固定已选来源后改变执行端所读表示，后置；它不是 E3 的别名。"
    "M1/M2/M3/M5 是受论文启发的同源表示控制，不是对那些论文完整系统的复现。"
    "历史条件的推测不等于经过独立反馈验证的反例。PRE 选择不能读取目标题 gold 或未来 gap。"
    "没有生成实验分数，未证明任何表示优越。"
)


def q(name: str) -> str:
    prefix, local = name.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def uid(slug: str) -> str:
    return f"urn:growrag:minimal-experiment:20260911:{slug}"


def add(parent: ET.Element, name: str, value=None, **attributes) -> ET.Element:
    node = ET.SubElement(parent, q(name), {q(k): v for k, v in attributes.items()})
    if value is not None:
        node.text = str(value)
    return node


def write_json(name: str, data) -> None:
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def note_sections(record: dict, note: dict) -> list[tuple[str, str]]:
    return [
        ("阅读用途与优先级", f"顺序 {note['order']:02d}；{GROUPS[note['group']]}。{note['role']}"),
        ("发表层级", note["level"]),
        ("重点读哪里", note["focus"]),
        ("借鉴什么", note["learn"]),
        ("已有覆盖与尚不能替代的实验", note["boundary"]),
        ("对本轮最小实验的约束", note["experiment"]),
        ("读完应能回答", note["question"]),
        ("全包实验定义", PROTOCOL),
        (
            "核验范围与使用说明",
            f"元数据核验日期：{record['verification_date']}；中文导读更新：{DATE}。"
            + record.get("verification_scope", "沿用先前书目快照与核验来源，本次未重新逐字核查该论文全文。")
            + " 导读由 AI 辅助整理，不是论文原摘要，也不是用户已读声明。"
            "预印本不等于未投稿；未统计引用数；无 PDF 附件；未向 Zotero 客户端导入。",
        ),
    ]


def render_note(record: dict, note: dict) -> str:
    parts = [f"<h2>{html.escape(note['label'])}｜2026-09-11 导读</h2>"]
    for title, body in note_sections(record, note):
        parts.append(f"<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p>")
    for title, link in [
        ("正式/书目入口", record["url"]),
        ("方法与数据阅读入口", note["method_url"]),
        ("元数据核验来源", record["verification_source"]),
    ]:
        parts.append(f'<p><a href="{html.escape(link, quote=True)}">{title}</a></p>')
    return "<div>" + "".join(parts) + "</div>"


def validate_records(records: list[dict]) -> None:
    for field in ("slug", "title"):
        values = [re.sub(r"\W+", "", str(record[field]).lower()) for record in records]
        assert len(values) == len(set(values)), f"Duplicate {field}"
    dois = [record["doi"].lower() for record in records if record.get("doi")]
    assert len(dois) == len(set(dois)), "Duplicate DOI"
    urls = [record["url"] for record in records]
    assert len(urls) == len(set(urls)), "Duplicate URL"
    for record in records:
        assert record["title"] and record["venue"] and record["authors"]
        assert re.fullmatch(r"\d{4}", str(record["year"]))
        assert record["type"] in {"conferencePaper", "journalArticle", "preprint"}
        for field in ("url", "verification_source"):
            parsed = urlparse(record[field])
            assert parsed.scheme == "https" and parsed.netloc
        if record.get("doi"):
            assert re.fullmatch(r"10\.\d{4,9}/\S+", record["doi"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", record["verification_date"])
        assert all(author["given"] and author["family"] for author in record["authors"])


def build_rdf(records: list[dict], plan: dict, filename: str, scope: str) -> dict:
    root = ET.Element(q("rdf:RDF"))
    for record in records:
        slug = record["slug"]
        note = plan[slug]
        item = add(root, "rdf:Description", **{"rdf:about": uid(slug)})
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
        # Preserve the existing project's tested RDF field pattern; validation
        # below is structural and does not claim a Zotero client import test.
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
        for tag in ["GrowRAG/最小表示选择实验/2026-09-11", GROUPS[note["group"]], note["level"], f"阅读顺序/{note['order']:02d}"]:
            add(item, "dc:subject", tag)
        add(item, "dcterms:isReferencedBy", **{"rdf:resource": uid(slug + ":note")})
        memo = add(root, "bib:Memo", **{"rdf:about": uid(slug + ":note")})
        add(memo, "rdf:value", render_note(record, note))
    overview_id = uid("overview:" + scope)
    overview = add(root, "bib:Memo", **{"rdf:about": overview_id})
    add(overview, "rdf:value", "<div><h2>本次阅读包用途</h2><p>" + html.escape(PROTOCOL)
        + "</p><p>先读 ReFormeR → AWM 相关小节 → ReMe → HotpotQA 数据与指标。"
        "RRM/QCR 是创新排重必读但仍按此前核验标为预印本；ExpeL 是重要补充。"
        "QPP系列、有限循环与结构分支后置，不要求读完17篇才做小实验。</p>"
        "<p>完整包17篇＝09-08聚焦12篇＋5篇补充。增量仅相对该12篇包，"
        "不是相对全部历史第一次推荐。不要同时重复导入完整和增量包。"
        "保留原有条目和批注；结构校验不等于Zotero客户端导入测试。</p></div>")
    parent = add(root, "z:Collection", **{"rdf:about": uid("collection:" + scope)})
    add(parent, "dc:title", f"GrowRAG｜最小表示选择实验｜{len(records)}篇｜2026-09-11")
    add(parent, "dcterms:hasPart", **{"rdf:resource": overview_id})
    active_groups = [group for group in GROUPS if any(plan[record["slug"]]["group"] == group for record in records)]
    for group in active_groups:
        collection_id = uid("collection:" + scope + ":" + group)
        add(parent, "dcterms:hasPart", **{"rdf:resource": collection_id})
        collection = add(root, "z:Collection", **{"rdf:about": collection_id})
        add(collection, "dc:title", GROUPS[group])
        for record in records:
            if plan[record["slug"]]["group"] == group:
                add(collection, "dcterms:hasPart", **{"rdf:resource": uid(record["slug"])})
    ET.indent(root, space="  ")
    path = ROOT / filename
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    parsed = ET.parse(path).getroot()
    identifiers = [element.attrib[q("rdf:about")] for element in parsed]
    assert len(identifiers) == len(set(identifiers))
    assert all(element.attrib[q("rdf:resource")] in identifiers for element in parsed.iter() if q("rdf:resource") in element.attrib)
    items = [element for element in parsed if element.find(q("z:itemType")) is not None]
    assert len(items) == len(records)
    for item, record in zip(items, records, strict=True):
        assert item.findtext(q("dc:title")) == record["title"]
        assert item.findtext(q("dc:date")) == str(record["year"])
        assert item.findtext(q("z:itemType")) == record["type"]
        assert item.findtext("dc:identifier/dcterms:URI/rdf:value", namespaces=NS) == record["url"]
        assert item.findtext("dcterms:isPartOf/bib:Journal/dc:title", namespaces=NS) == record["venue"]
        actual_people = [{"given": person.findtext(q("foaf:givenName")), "family": person.findtext(q("foaf:surname"))}
                         for person in item.findall("bib:authors/rdf:Seq/rdf:li/foaf:Person", NS)]
        assert actual_people == record["authors"], f"Author order mismatch: {record['slug']}"
        actual_doi = item.findtext("dcterms:isPartOf/bib:Journal/dc:identifier", namespaces=NS)
        assert actual_doi == ("DOI " + record["doi"] if record.get("doi") else None)
        assert len(item.findall(q("dcterms:isReferencedBy"))) == 1
    memos = parsed.findall(q("bib:Memo"))
    assert len(memos) == len(records) + 1
    for memo in memos:
        ET.fromstring(memo.findtext(q("rdf:value")))
    collection_count = len(parsed.findall(q("z:Collection")))
    assert collection_count == len(active_groups) + 1
    return {
        "file": filename,
        "papers": len(items),
        "paper_child_notes": len(records),
        "notes_including_overview": len(memos),
        "collections": collection_count,
        "types": dict(Counter(record["type"] for record in records)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duplicate_ids": 0,
        "duplicate_dois": 0,
        "duplicate_titles": 0,
        "dangling_relations": 0,
        "author_order_doi_url_metadata_roundtrip": "passed",
        "child_note_html_parse": "passed",
        "zotero_client_import_tested": False,
        "pdf_attachments": 0,
    }


def write_readme(records: list[dict], plan: dict) -> None:
    lines = [
        "# GrowRAG｜最小表示选择实验阅读包｜2026-09-11", "",
        "这次不是增加研究方向，而是为下一轮最小实验重新排序。方向一：怎样选到值得执行的改写（主任务）；方向二：经验该保留哪些适用信息（配套）。下面的分组只是阅读用途。", "",
        "## 先下载哪一个", "",
        f"- [完整17篇 RDF]({FULL_NAME})：旧12篇＋Adaptive-RAG、HotpotQA、DocBench、DocLayNet、DecompRC，并为全部17篇重写中文子笔记。",
        f"- [相对09-08聚焦包补充5篇 RDF]({INCREMENT_NAME})：仅相对于 `2026-09-08_memory_cards` 的12篇包增加5篇，不表示这些都是全部历史里首次推荐。HotpotQA等已有历史推荐。",
        "- 已导入旧12篇者：可只导入补充5篇，旧12篇的新导读在本页和 notes/ 中。完整包与增量包不要重复导入；跨历史包 Zotero 仍可能出现重复条目，应保留已有批注后谨慎合并。",
        "- Zotero 可用“文件 → 导入”选择 RDF。这里只生成本地文件，没有操作 Zotero 客户端，也没有下载 PDF。", "",
        "## 最短阅读顺序", "",
        "**ReFormeR → AWM 的 §4 与偏离流程讨论 → ReMe → HotpotQA 的数据与指标。**", "",
        "先理解模式来自什么、如何做表示对照、场景条件怎样组织，最后明确数据输入和评分。无需读完17篇才执行开发性小实验。", "",
        "随后 RRM → QCR 做创新排重：二者仍按此前核验标为预印本，但与我们足够接近，正式写创新之前必须读。ExpeL 是具体经验与洞见互补的重要补充。", "",
        "后续增强按用途读：QPP → ReformIR → RAQG-QPP；充分性/有限循环再读 Sufficient Context → S2G-RAG → GAM-RAG；结构分支按需读 Adaptive-RAG → DecompRC → DocBench → DocLayNet。", "",
        "## 三个实验名称不要混淆", "",
        "| 实验 | 固定什么、改变什么 | 本次安排 |",
        "| --- | --- | --- |",
        "| E1：更会选择吗 | 固定来源、候选动作、当前输入、选择器、规范执行；只改变选择器所读经验表示 | 首轮 |",
        "| E2：更会应用吗 | 固定已选经验来源，改变执行器所读经验表示 | 延后 |",
        "| E3：经验有没有额外收益 | 在共同底座与预算协议下比较 REUSE 与仅现场 FRESH；不是与完全不修复相比就把收益归给经验 | 随首轮报告 |", "",
        "M1/M2/M3/M5 等表示是论文启发下的局部控制，不是完整系统复现。优质摘要也可以包含条件；结构化和自然语言若信息量不同，不能把差异都归因于格式。新条件推测不是已验证反例。PRE 不看目标题金标/未来缺口。未执行候选的真实结果不能由模型猜测补齐。", "",
        "## 发表层级与核验边界", "",
        "17篇中：14篇会议/会议论文集条目（其中 DocBench 是 workshop，ReMe 是 Findings），1篇 TOIS 期刊，2篇预印本。ECIR是重要检索会议，但不在这里与ICML/ACL等简单混为同一档。", "",
        "原12篇元数据逐项保留原 verification_date 和 verification_source，不声称 2026-09-11 重新核验了所有全文或所有发表状态。新增5篇本次核查官方书目来源；方法入口有作者公开版本和仓库，阅读范围见 metadata_snapshot.json。预印本不等于未投稿；不凭空报引用数。", "",
        "这是一份 AI 辅助导读和实验计划索引，不是已有实验结果，不代表用户已读，也不能保证研究绝对新颖。", "",
        "## 每篇读什么", "",
    ]
    for record in records:
        note = plan[record["slug"]]
        lines += [f"### {note['order']:02d}. {note['label']}", "", f"[{record['title']}]({record['url']})", ""]
        for title, body in note_sections(record, note):
            if title == "全包实验定义":
                continue
            lines += [f"- {title}：{body}"]
        lines += [f"- [方法/数据入口]({note['method_url']})；[独立导读](notes/{record['slug']}.md)。", ""]
    lines += ["## 验证与复现", "", "生成脚本只离线读取旧快照与本目录输入，输出限本目录；无网络、无 API、无业务代码修改。validation.json 记录 XML、作者顺序、DOI/URL回读、去重、子笔记和关联验证及文件哈希；结构通过不代表已在 Zotero 界面导入测试。", ""]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    notes_dir = ROOT / "notes"
    notes_dir.mkdir(exist_ok=True)
    for record in records:
        note = plan[record["slug"]]
        note_lines = [f"# {note['label']}｜2026-09-11 导读", "", f"[{record['title']}]({record['url']})", ""]
        for title, body in note_sections(record, note):
            note_lines += [f"## {title}", "", body, ""]
        note_lines += [f"[方法/数据入口]({note['method_url']})；[元数据核验来源]({record['verification_source']})。", ""]
        (notes_dir / (record["slug"] + ".md")).write_text("\n".join(note_lines), encoding="utf-8")


def main() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    additions = json.loads((ROOT / "metadata_new.json").read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "reading_plan.json").read_text(encoding="utf-8"))
    assert len(previous) == 12 and len(additions) == 5 and len(plan) == 17
    previous_ids = {record["slug"] for record in previous}
    new_ids = {record["slug"] for record in additions}
    assert previous_ids.isdisjoint(new_ids)
    assert new_ids == {"adaptive_rag", "hotpotqa", "docbench", "doclaynet", "decomprc"}
    lookup = {record["slug"]: copy.deepcopy(record) for record in previous + additions}
    assert set(lookup) == set(plan)
    order = sorted(plan, key=lambda slug: plan[slug]["order"])
    assert [plan[slug]["order"] for slug in order] == list(range(1, 18))
    records = [lookup[slug] for slug in order]
    for old in previous:
        assert lookup[old["slug"]] == old, "Prior metadata must remain unchanged"
    validate_records(records)
    increments = [record for record in records if record["slug"] in new_ids]
    reports = [build_rdf(records, plan, FULL_NAME, "full"), build_rdf(increments, plan, INCREMENT_NAME, "increment")]
    write_json("metadata_snapshot.json", records)
    write_readme(records, plan)
    report = {
        "built_date": DATE,
        "scope": "Offline Zotero bibliography/readme assets only; not runtime code or experiments",
        "files": reports,
        "original_12_metadata_preserved_exactly": True,
        "increment_relative_to": "2026-09-08_memory_cards (12 papers), not all historical recommendations",
        "increment_papers": [record["slug"] for record in increments],
        "increment_count": 5,
        "new_metadata_official_sources_checked_date": DATE,
        "all_17_full_texts_rechecked_today": False,
        "network_checks_in_build": False,
        "per_paper_markdown_notes": len(list((ROOT / "notes").glob("*.md"))),
        "publication_caveats": ["DocBench: workshop", "ReMe: Findings", "RRM/QCR: preprint status retained from prior verification"],
        "zotero_client_import_tested": False,
    }
    assert report["per_paper_markdown_notes"] == 17
    write_json("validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
