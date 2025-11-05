#!/usr/bin/env python3
from __future__ import annotations

"""
Selective downloader for arXiv LaTeX tarballs (math categories).

Defaults to saving under ProjectSearchBar/data/papers, but you can continue to use
your existing `/home/Brandon/arxiv_tex_selected` without moving files.

Usage examples:
  python3 tools/download.py --out ./data/papers --max 50000 --from 2010-01-01
"""

import argparse
import time
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
import requests

from ProjectSearchBar import config


UA = {"User-Agent": "ProjectSearchBar/1.0 (mailto:your-email@example.com)"}
OAI_ENDPOINT = "https://export.arxiv.org/oai2"
ARXIV_ATOM = "http://export.arxiv.org/api/query"
EPRINT_BASE = "https://arxiv.org/e-print"

NS_OAI = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arXiv": "http://arxiv.org/OAI/arXiv/"
}
NS_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def oai_listrecords_math(resumption_token=None, from_date=None, until_date=None, set_name="math"):
    params = {"verb": "ListRecords"}
    if resumption_token:
        params["resumptionToken"] = resumption_token
    else:
        params.update({"metadataPrefix": "arXiv", "set": set_name})
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date

    r = requests.get(OAI_ENDPOINT, params=params, headers=UA, timeout=60)
    r.raise_for_status()

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        print("OAI non-XML response snippet:\n", r.text[:500])
        return [], None

    out = []
    for rec in root.findall(".//oai:record", NS_OAI):
        header = rec.find("oai:header", NS_OAI)
        if header is not None and header.get("status") == "deleted":
            continue
        ident = header.findtext("oai:identifier", default="", namespaces=NS_OAI) if header is not None else ""
        arx_id = ident.split(":")[-1] if ident else ""
        if arx_id:
            out.append(arx_id)
    token_el = root.find(".//oai:resumptionToken", NS_OAI)
    next_token = token_el.text.strip() if (token_el is not None and token_el.text) else None
    return out, next_token


def safe_extract_tar(tar_path: Path, dest_dir: Path):
    with tarfile.open(tar_path, "r:*") as tf:
        base = dest_dir.resolve()
        for m in tf.getmembers():
            p = (dest_dir / m.name).resolve()
            if not str(p).startswith(str(base)):
                continue
            tf.extract(m, dest_dir)


def download_tex_tarball(base_id: str, outdir: Path) -> str:
    url = f"{EPRINT_BASE}/{base_id}"
    out_tar = outdir / f"{base_id.replace('/', '_')}.tar.gz"
    if out_tar.exists() and out_tar.stat().st_size > 0:
        return "skip"
    r = requests.get(url, headers=UA, stream=True, timeout=180)
    if r.status_code != 200:
        return f"err:{r.status_code}"
    with open(out_tar, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download arXiv LaTeX tarballs (math)")
    ap.add_argument('--out', type=Path, default=config.DATA_DIR / 'papers')
    ap.add_argument('--max', type=int, default=50000)
    ap.add_argument('--from', dest='from_date', type=str, default='2010-01-01')
    ap.add_argument('--until', dest='until_date', type=str, default=None)
    ap.add_argument('--sleep', type=float, default=3.0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    token = None
    total = 0
    pages = 0
    while total < args.max:
        ids, token = oai_listrecords_math(resumption_token=token, from_date=args.from_date, until_date=args.until_date)
        pages += 1
        if not ids:
            break
        for bid in ids:
            if total >= args.max:
                break
            status = download_tex_tarball(bid, args.out)
            if status == 'ok':
                total += 1
            if total % 25 == 0:
                print(f"{total} downloaded...")
            time.sleep(args.sleep)
        if not token:
            break
        time.sleep(args.sleep)
    print(f"Done. Downloaded: {total}. Saved to: {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
