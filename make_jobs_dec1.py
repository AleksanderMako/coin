"""Write the job lists for the 2026-12-01 experiments. Usage: python make_jobs_dec1.py --out jobs [--c4_iters N]

  dec1_b_{1..4}.txt     B: 132 timed fits on the 24 Kodak images, split by image over four A40 pods
  dec1_c3_{1..4}.txt    C3: 48 frozen-first-layer fits, same image split (queued behind B)
  dec1_a_{1,2}.txt      A: 108 QAT runs (36 fits x 3 arms), 54 per 3090 pod, fits uploaded to runs/fits/<tag>
  dec1_c4_{1,2}.txt     C4: 100 DIV2K fits (0801 to 0900), 50 per 3090 pod
  dec1_qat_ref.txt      the A40 reference QAT run (good rule, lambda 0.1, k_lbfgs_s1_im23)
  fits_dec1.list        the 36 local fit paths (no extension) to upload as runs/fits/<basename>.{pt,json}

Every command carries --tag (launch.py resumes on <out>/<tag>.json) and --out runs; one job per card.
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--out", default="jobs")
p.add_argument("--c4_iters", type=int, default=5000,
               help="L-BFGS iterations for C4: the median of B's warm-L-BFGS arm at 360 s on the A40")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

E = "python experiment.py"
Q = "python qat.py"
HELD_OUT = [1, 5, 8, 14, 20, 23]  # the learned starts never saw these
IMAGES = list(range(1, 25))
STARTS = {"lbfgs": "runs/night1/reptile_lbfgs.pt", "adam": "runs/night1/reptile_adam.pt"}
FIT_DIR = "runs/fits"
OUT = "--out runs"

ADAM50K = "--optimizer adam --num_iters 50000"
ADAM10 = "--optimizer adam --num_iters 50000 --max_seconds 600"
LBFGS_WARM10 = "--optimizer lbfgs --warmup_iters 500 --history 100 --num_iters 100000 --max_seconds 600"
LBFGS_COLD10 = "--optimizer lbfgs --warmup_iters 0 --history 100 --num_iters 100000 --max_seconds 600"
SH_ADAM, SH_LBFGS = "--sharpness_every 500", "--sharpness_every 50"


def write(name, lines):
    with open(os.path.join(args.out, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} jobs")


def fit(im, arm, flags, sharp, seed=0, extra=""):
    return f"{E} --dataset kodak --image {im} --crop 1 {flags}{extra} --seed {seed} {sharp} --tag d1_{arm}_s{seed}_im{im} {OUT}"


def b_jobs(im):
    jobs = [fit(im, "adam50k", ADAM50K, SH_ADAM),
            fit(im, "adamsol", ADAM10 + " --solve_last", SH_ADAM),
            fit(im, "lbfgswarm", LBFGS_WARM10, SH_LBFGS),
            fit(im, "lbfgssol", LBFGS_COLD10 + " --solve_last", SH_LBFGS)]
    if im in HELD_OUT:
        jobs += [fit(im, "adamsol", ADAM10 + " --solve_last", SH_ADAM, seed=1),
                 fit(im, "lbfgssol", LBFGS_COLD10 + " --solve_last", SH_LBFGS, seed=1)]
        # an --init arm has no seed (the start is loaded, the fit is deterministic): the replication is the second start
        for which, path in STARTS.items():
            jobs += [fit(im, f"adaminit{which}", ADAM10, SH_ADAM, extra=f" --init {path}"),
                     fit(im, f"lbfgsinit{which}", LBFGS_COLD10, SH_LBFGS, extra=f" --init {path}")]
    return jobs


def c3_jobs(im):
    return [fit(im, "adamfrz", "--optimizer adam --num_iters 50000 --max_seconds 360 --freeze_first", SH_ADAM),
            fit(im, "lbfgsfrz", "--optimizer lbfgs --warmup_iters 500 --history 100 --num_iters 100000 --max_seconds 360 "
                                "--freeze_first", SH_LBFGS)]


# B and C3: image i goes to pod (i - 1) % 4 + 1, so each pod has six images, six Adam-50k jobs,
# and one or two of the held-out images (pods 1 and 4 get two, pods 2 and 3 one)
for k in range(1, 5):
    ims = [im for im in IMAGES if (im - 1) % 4 + 1 == k]
    write(f"dec1_b_{k}.txt", [j for im in ims for j in b_jobs(im)])
    write(f"dec1_c3_{k}.txt", [j for im in ims for j in c3_jobs(im)])

# A: the 36 saved fits, original tags kept; uploaded flat to runs/fits/<tag>.{pt,json}
seed0_adam = {im: ("f_adam_im%d" if im in (1, 8, 23) else "g_adam_im%d") % im for im in HELD_OUT}
seed0_lbfgs = {im: ("f_lbfgs_im%d" if im in (1, 8, 23) else "g_lbfgs_cold_im%d") % im for im in HELD_OUT}
local = []
for im in HELD_OUT:
    local += [f"runs/{seed0_adam[im]}", f"runs/{seed0_lbfgs[im]}",
              f"runs/night1/k_adam_s1_im{im}", f"runs/night1/k_lbfgs_s1_im{im}",
              f"runs/night1/llbfgs_adam_im{im}", f"runs/night1/llbfgs_lbfgs_cold_im{im}"]
write("fits_dec1.list", local)  # not .txt: it is a file list, not a job list

ARMS = [("good", "0"), ("good", "0.1"), ("crude", "0.1")]


def qat(fit_tag, rule, lam, tag=None):
    return (f"{Q} --fit {FIT_DIR}/{fit_tag} --rule {rule} --bits 8 --lam {lam} --iters 15000 --lr 2e-4 "
            f"--probes 8 --power_iters 20 --sharpness_every 0 --tag {tag or f'd1_qat_{rule}_lam{lam}_{fit_tag}'} {OUT}")


a = {1: [], 2: []}
for n, path in enumerate(local):  # alternating fits: each pod gets every start type, optimiser and image
    a[1 + n % 2] += [qat(os.path.basename(path), rule, lam) for rule, lam in ARMS]
write("dec1_a_1.txt", a[1])
write("dec1_a_2.txt", a[2])
write("dec1_qat_ref.txt", [qat("k_lbfgs_s1_im23", "good", "0.1", tag="d1_qat_ref")])

# C4: DIV2K validation 0801..0900, warm L-BFGS budgeted by iterations, alternating over two pods
c4 = {1: [], 2: []}
for im in range(801, 901):
    c4[1 + (im + 1) % 2].append(f"{E} --dataset div2k --image {im} --crop 1 --optimizer lbfgs --warmup_iters 500 --history 100 "
                           f"--num_iters {args.c4_iters} --seed 0 --sharpness_every 0 --tag d1_c4lbfgs_s0_im{im} {OUT}")
write("dec1_c4_1.txt", c4[1])
write("dec1_c4_2.txt", c4[2])
