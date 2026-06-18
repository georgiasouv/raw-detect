"""One-time dataset prep: patch COCO JSONs, quarantine corrupt RAW, pre-pack to .npy."""
import argparse, glob, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from rawdet.packing import pack_rggb, normalize_raw


def patch_json(ann_dir):
    for p in glob.glob(os.path.join(ann_dir, '*.json')):
        d = json.load(open(p))
        if 'info' not in d or 'licenses' not in d:
            d.setdefault('info', {}); d.setdefault('licenses', [])
            json.dump(d, open(p, 'w'))
            print(f"  patched info/licenses -> {os.path.basename(p)}")


def pack_one(args):
    src, dst = args
    if os.path.exists(dst):
        return (src, 'skip')
    try:
        import rawpy
        with rawpy.imread(src) as raw:
            bayer = raw.raw_image_visible.astype(np.float32)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
        packed = pack_rggb(normalize_raw(bayer, black, white))
        np.save(dst, packed.astype(np.float16))
        return (src, 'ok')
    except Exception as e:
        return (src, f'FAIL: {str(e)[:50]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-glob', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ann-dir', default=None)
    ap.add_argument('--corrupt', default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    if args.ann_dir:
        print("patching annotation JSONs ...")
        patch_json(args.ann_dir)

    skip = set()
    if args.corrupt and os.path.exists(args.corrupt):
        skip = {l.strip() for l in open(args.corrupt) if l.strip()}
        print(f"will skip {len(skip)} known-corrupt files")

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.raw_glob))
    jobs = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if os.path.basename(f) in skip:
            continue
        jobs.append((f, os.path.join(args.out_dir, stem + '.npy')))
    print(f"packing {len(jobs)} of {len(files)} files ({len(files)-len(jobs)} skipped) ...")

    ok = bad = skipped = done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(pack_one, j) for j in jobs]):
            src, status = fut.result(); done += 1
            if status == 'ok': ok += 1
            elif status == 'skip': skipped += 1
            else: bad += 1; print(f"  {os.path.basename(src)}: {status}")
            if done % 500 == 0: print(f"  {done}/{len(jobs)}  (ok={ok} skip={skipped} fail={bad})")
    print(f"\nDONE: {ok} packed, {skipped} present, {bad} failed -> {args.out_dir}")


if __name__ == '__main__':
    main()
