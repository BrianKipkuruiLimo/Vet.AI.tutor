#!/usr/bin/env python3
import json
import os
import zipfile
import shutil
import tempfile
import sys

REPORT = os.path.join('archive','inspect_raw_notes_report.json')
OUT = os.path.join('archive','repair_results.json')
PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x1a\xe6\xb6\x00\x00\x00\x00IEND\xaeB`\x82"
)


def repair_zip_file(abs_path):
    print(f"\n== Repairing: {abs_path}")
    if not os.path.exists(abs_path):
        print(" -> file missing")
        return False, "missing"

    try:
        with zipfile.ZipFile(abs_path, 'r') as z:
            names = z.namelist()
    except Exception as e:
        print(" -> cannot open archive:", repr(e))
        return False, f"open_failed:{e!r}"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.zip')
    os.close(tmp_fd)
    bad_members = []
    try:
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            with zipfile.ZipFile(abs_path, 'r') as zin:
                for name in names:
                    try:
                        data = zin.read(name)
                        zout.writestr(name, data)
                    except Exception as e:
                        print("  - skipping bad member:", name, "err:", repr(e))
                        bad_members.append(name)
                        # For media files, try writing a 1x1 placeholder PNG so document still opens
                        lname = name.lower()
                        if lname.endswith(('.png', '.jpg', '.jpeg')):
                            try:
                                zout.writestr(name, PLACEHOLDER_PNG)
                                print("    -> wrote placeholder for", name)
                            except Exception as e2:
                                print("    -> placeholder write failed:", e2)
                        # otherwise omit the member
        # Validate repaired archive
        try:
            with zipfile.ZipFile(tmp_path, 'r') as z2:
                bad = z2.testzip()
                if bad is None:
                    # backup original and replace
                    backup = abs_path + '.bak'
                    shutil.move(abs_path, backup)
                    shutil.move(tmp_path, abs_path)
                    print("  -> repaired successfully; original backed up to", backup)
                    return True, {'bad_members': bad_members}
                else:
                    print("  -> repaired archive still has bad member:", bad)
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                    return False, f"still_bad:{bad}"
        except Exception as e:
            print("  -> validation failed:", e)
            try:
                os.remove(tmp_path)
            except:
                pass
            return False, f"validation_failed:{e!r}"
    except Exception as e:
        print("  -> error while building repaired archive:", e)
        try:
            os.remove(tmp_path)
        except:
            pass
        return False, f"build_error:{e!r}"


def main():
    if not os.path.exists(REPORT):
        print('Report not found:', REPORT)
        return 2
    with open(REPORT, 'r', encoding='utf-8') as f:
        rep = json.load(f)

    targets = []
    for e in rep.get('files', []):
        path = e.get('path')
        if not path:
            continue
        # focus on zip-based containers
        container = e.get('container')
        typ = e.get('type')
        # consider entries flagged as zip errors or testzip bad (but skip skipped_large)
        if e.get('zip_error') or (e.get('zip_testzip_bad') and e.get('zip_testzip_bad') != 'skipped_large'):
            # prefer docx/pptx or type zip
            if container in ('docx','pptx') or typ == 'zip' or path.lower().endswith(('.docx','.pptx')):
                targets.append(path)

    if not targets:
        print('No repair targets found.')
        return 0

    results = {}
    cwd = os.getcwd()
    for rel in targets:
        # normalize path
        if os.path.isabs(rel):
            p = rel
        else:
            p = os.path.normpath(os.path.join(cwd, rel))
        ok, msg = repair_zip_file(p)
        results[rel] = {'ok': ok, 'msg': msg}

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\nWrote repair results to', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
