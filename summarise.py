"""One line per run JSON: steps, ms per step, quality, sharpness, rounding. Usage: python summarise.py runs/*.json"""
import json
import sys



def num(x, nd=1):
    return "-" if x is None or x != x else f"{round(x, nd):g}"


rows = []
for f in sys.argv[1:]:
    d = json.load(open(f))
    c = d["config"]
    q = d["psnr_after_rounding"]
    ms = 1000 * d["fit_seconds"] / max(d["steps"] - 1, 1)
    sharp = d["log"].get("sharpness", [])
    along = " ".join(num(s['lambda_max']) for s in sharp[-4:]) if sharp else "-"
    rows.append((c.get("lr", 0), f"{c['tag']:28} {c['optimizer']:5} lr {c['lr']:<8.2g} 2/lr {2 / c['lr']:<9.3g} "
                 f"steps {d['steps']:6d} {ms:6.1f} ms/step {d['fit_seconds']:7.1f}s psnr {d['final_psnr']:6.2f} "
                 f"{'DIV ' if d.get('diverged') else '    '}lam {num(d['lambda_max']):>9} tr {num(d['hessian_trace'], 0):>8} "
                 f"8bit {q['uniform8'] - q['fp32']:+6.2f} ch8 {q['channel8'] - q['fp32']:+6.2f} "
                 f"8b/f16 {q['uniform8_first16'] - q['fp32']:+6.2f}  lam along: {along}  [{d.get('gpu', '?')}]"))
for _, r in sorted(rows, key=lambda x: (x[0], x[1])):
    print(r)
