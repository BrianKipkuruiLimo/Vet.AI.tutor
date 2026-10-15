#!/usr/bin/env python3
import json
import zipfile
from pathlib import Path
from collections import Counter

ROOT = Path("data/raw_notes")
ZIP_TEST_THRESHOLD = 50 * 1024 * 1024  # 50 MB

report = {"files": []}

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    rec = {"path": str(path), "size": path.stat().st_size}
    try:
        with path.open("rb") as f:
            sample = f.read(1024)
    except Exception as e:
        rec["error"] = f"read_failed: {e}"
        report["files"].append(rec)
        continue

    rec["header_hex"] = " ".join(f"{b:02X}" for b in sample[:64])
    try:
        rec["preview_text"] = sample.decode("utf8", errors="replace")[:256]
    except Exception:
        rec["preview_text"] = None

    if b"version https://git-lfs.github.com/spec/v1" in sample:
        rec["type"] = "git-lfs-pointer"

    elif sample.lstrip().startswith(b"%PDF"):
        rec["type"] = "pdf"
        try:
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 2048))
                tail = f.read(2048)
            rec["has_eof"] = b"%%EOF" in tail
        except Exception as e:
            rec["tail_error"] = str(e)

    elif sample.startswith(b"PK"):
        rec["type"] = "zip"
        try:
            with zipfile.ZipFile(path, "r") as z:
                nl = z.namelist()
                rec["zip_namelist_count"] = len(nl)
                rec["zip_names_preview"] = nl[:10]
                if any(name.startswith("ppt/") for name in nl):
                    rec["container"] = "pptx"
                elif any(name.startswith("word/") for name in nl):
                    rec["container"] = "docx"
                else:
                    rec["container"] = "zip"

                if path.stat().st_size <= ZIP_TEST_THRESHOLD:
                    bad = z.testzip()
                    rec["zip_testzip_bad"] = bad
                else:
                    rec["zip_testzip_bad"] = "skipped_large"
        except zipfile.BadZipFile as e:
            rec["zip_error"] = f"BadZipFile: {e}"
        except Exception as e:
            rec["zip_error"] = str(e)

    else:
        rec["type"] = "unknown"

    report["files"].append(rec)

out = Path("archive/inspect_raw_notes_report.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf8")

types = Counter(f.get("type", "unknown") for f in report["files"])
print("Summary counts:", dict(types))
print("Problematic entries (samples):")
cnt = 0
for f in report["files"]:
    bad_cond = (
        f.get("type") == "git-lfs-pointer"
        or f.get("zip_error")
        or (f.get("type") == "zip" and f.get("zip_testzip_bad") and f.get("zip_testzip_bad") != "skipped_large")
        or (f.get("type") == "pdf" and not f.get("has_eof", True))
    )
    if bad_cond:
        cnt += 1
        print(f"{f['path']} size={f['size']} type={f.get('type')} msg={f.get('zip_error') or f.get('zip_testzip_bad') or ('missing_eof' if f.get('type')=='pdf' and not f.get('has_eof') else '')}")
        if cnt >= 50:
            break

print(f"Report written to {out}")
