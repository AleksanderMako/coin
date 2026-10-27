"""Write the job lists for the overnight pods. Usage: python make_jobs.py --gd_lrs 3e-3 1e-2 3e-2 1e-1 2e-1 --out jobs/
One file per pod; every command carries a --tag so launch.py can resume. One fit at a time per card
(the smoke test showed concurrent fits share the card with no throughput gain).

  sweep_1.txt, sweep_2.txt  learning-rate sweep on half crops, sharpness logged, one L4 each
  finish_then_seeds.txt     flat-seeking finish from the saved seed-zero fits, then the second seed (timed), A40
  reptile_adam.txt          meta-learn a start with an Adam inner loop, then fits from it on the test images (timed), A40
  reptile_lbfgs.txt         same with an L-BFGS inner loop, A40
  baselines.txt             cold L-BFGS from random at the learned-start budget, appended to the seeds pod
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--gd_lrs", type=float, nargs="+", required=True)
p.add_argument("--adam_lrs", type=float, nargs="+", default=[1e-4, 2e-4, 5e-4])
p.add_argument("--sweep_images", type=int, nargs="+", default=[1, 5, 8, 14, 20, 23])
p.add_argument("--test_images", type=int, nargs="+", default=[1, 5, 8, 14, 20, 23])
p.add_argument("--sweep_iters", type=int, default=50000)
p.add_argument("--lbfgs_seconds", type=int, default=600)
p.add_argument("--learned_seconds", type=int, default=360)
p.add_argument("--rhos", type=float, nargs="+", default=[0.02, 0.1])
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--out", default="jobs")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
train_images = [i for i in range(1, 25) if i not in args.test_images]
E = "python experiment.py"


def write(name, lines):
    with open(os.path.join(args.out, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} jobs")


# the learning-rate sweep (link six), split over the two L4s by image
sweep = {1: [], 2: []}
for n, im in enumerate(args.sweep_images):
    part = sweep[1 + n % 2]
    for lr in args.gd_lrs:
        part.append(f"{E} --image {im} --crop 2 --optimizer gd --lr {lr:g} --num_iters {args.sweep_iters} "
                    f"--sharpness_every 500 --tag s_gd_lr{lr:g}_im{im}")
    for lr in args.adam_lrs:
        part.append(f"{E} --image {im} --crop 2 --optimizer adam --lr {lr:g} --num_iters {args.sweep_iters} "
                    f"--sharpness_every 500 --tag s_adam_lr{lr:g}_im{im}")
write("sweep_1.txt", sweep[1])
write("sweep_2.txt", sweep[2])

# flat-seeking finish from the saved seed-zero fits (links five and seven), short, then the timed second seed
jobs = []
saved = {1: ("f_adam_im1", "f_lbfgs_im1"), 8: ("f_adam_im8", "f_lbfgs_im8"), 23: ("f_adam_im23", "f_lbfgs_im23"),
         5: ("g_adam_im5", "g_lbfgs_cold_im5"), 14: ("g_adam_im14", "g_lbfgs_cold_im14"),
         20: ("g_adam_im20", "g_lbfgs_cold_im20")}
for im, tags in saved.items():
    for src in tags:
        for rho in args.rhos:
            jobs.append(f"{E} --image {im} --crop 1 --optimizer adam --num_iters 0 --init runs/{src}.pt "
                        f"--sam_steps 500 --sam_rho {rho:g} --sharpness_every 100 --tag fin_{src}_rho{rho:g}")
        # control: the same five hundred Adam steps with no perturbation
        jobs.append(f"{E} --image {im} --crop 1 --optimizer adam --num_iters 500 --init runs/{src}.pt "
                    f"--sharpness_every 100 --tag fin_{src}_plain")
for im in args.test_images:
    jobs.append(f"{E} --image {im} --crop 1 --optimizer adam --num_iters 50000 --seed 1 --tag k_adam_s1_im{im}")
    jobs.append(f"{E} --image {im} --crop 1 --optimizer lbfgs --warmup_iters 500 --history 100 "
                f"--num_iters 100000 --max_seconds {args.lbfgs_seconds} --seed 1 --tag k_lbfgs_s1_im{im}")
# baseline for the learned-start figure on the same card type: cold L-BFGS from random at the same budget
for im in args.test_images:
    jobs.append(f"{E} --image {im} --crop 1 --optimizer lbfgs --warmup_iters 0 --history 100 --num_iters 100000 "
                f"--max_seconds {args.learned_seconds} --tag base_lbfgs_cold_im{im}")
write("finish_then_seeds.txt", jobs)

# the learned start (link eight): meta-train, then timed fits from it on the held-out images
imgs = " ".join(map(str, train_images))
for which, inner in (("adam", "--inner adam --inner_steps 500"), ("lbfgs", "--inner lbfgs --inner_steps 100")):
    init = f"runs/reptile_{which}.pt"
    jobs = [f"python reptile.py --images {imgs} {inner} --epochs {args.epochs} --meta_lr 0.5 --out {init} "
            f"--tag reptile_{which}"]
    for im in args.test_images:
        jobs.append(f"{E} --image {im} --crop 1 --optimizer lbfgs --warmup_iters 0 --history 100 --num_iters 100000 "
                    f"--max_seconds {args.learned_seconds} --init {init} --tag l{which}_lbfgs_cold_im{im}")
        jobs.append(f"{E} --image {im} --crop 1 --optimizer lbfgs --warmup_iters 500 --history 100 --num_iters 100000 "
                    f"--max_seconds {args.learned_seconds} --init {init} --tag l{which}_lbfgs_warm_im{im}")
        jobs.append(f"{E} --image {im} --crop 1 --optimizer adam --num_iters 50000 "
                    f"--max_seconds {args.learned_seconds} --init {init} --tag l{which}_adam_im{im}")
    write(f"reptile_{which}.txt", jobs)
