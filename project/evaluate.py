import argparse
import json
import os
import time
from importlib import import_module
from multiprocessing import Pool

import numpy as np
import pcg_benchmark

from run import convert2Dic

# Level 3 evaluation harness.
# Every generator gets the same N runs; run i uses seed (base_seed + i) and a control target sampled
# from that seed, so all generators are asked for exactly the same targets. The best level of each run
# goes into one set per generator, and the set is scored with the benchmark's quality, diversity and
# controlability. Diversity is only meaningful over a set, which is why it is measured here and not per run.

def to_python(value):
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_python(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def sample_control(problem, seed):
    env = pcg_benchmark.make(problem)
    env.seed(seed)
    return to_python(env.control_space.sample())

def run_once(job):
    problem, generator_name, seed, steps, early_stop, gen_args, folder = job
    env = pcg_benchmark.make(problem)
    env.seed(seed)
    control = sample_control(problem, seed)

    generator = import_module(f"generators.{generator_name}").Generator(env)
    start = time.time()
    generator.reset(seed=seed, steps=steps, control=control, **gen_args)
    history = [generator.stats() if hasattr(generator, "stats") else {"best": generator.best()}]
    for _ in range(steps):
        generator.update()
        history.append(generator.stats() if hasattr(generator, "stats") else {"best": generator.best()})
        if early_stop and generator.best() >= 1:
            break
    # learned generators keep their trained model (the "brain"), e.g. the Q-table
    if getattr(generator, "saves_brain", False):
        generator.save(os.path.join(folder, f"brain_{seed}"))

    return {"generator": generator_name, "seed": seed, "control": control,
            "content": to_python(generator.best_content()), "history": history,
            "iterations": len(history) - 1, "time": time.time() - start}

def run_single(problem, generator_name, seeds, steps, workers, gen_args):
    # one long run whose result serves every target: for generators like MAP-Elites, where a single archive
    # holds a level for each control target. Uses the same targets (one per seed) as the per-run mode.
    controls = [sample_control(problem, seed) for seed in seeds]
    env = pcg_benchmark.make(problem)
    env.seed(seeds[0])
    generator = import_module(f"generators.{generator_name}").Generator(env)
    start = time.time()
    generator.reset(seed=seeds[0], steps=steps, control=controls[0], workers=workers, **gen_args)
    history = [generator.stats()]
    for i in range(steps):
        generator.update()
        history.append(generator.stats())
        if (i + 1) % 25 == 0:
            s = history[-1]
            print(f"  step {i+1}/{steps}: " + ", ".join(f"{k} {v:.3f}" for k, v in s.items() if isinstance(v, float))
                  + f", {time.time() - start:.0f}s", flush=True)
    elapsed = time.time() - start

    runs = []
    for seed, control in zip(seeds, controls):
        generator.control = control
        runs.append({"generator": generator_name, "seed": seed, "control": control,
                     "content": to_python(generator.best_content()), "best": generator.best(),
                     "history": history if seed == seeds[0] else [],
                     "iterations": steps, "time": elapsed})
    return generator, runs

def score_set(problem, runs):
    env = pcg_benchmark.make(problem)
    contents = [r["content"] for r in runs]
    controls = [r["control"] for r in runs]
    q_score, d_score, c_score, details, _ = env.evaluate(contents, controls)
    return {
        # mean of the continuous [0, 1] values
        "quality": float(np.mean(details["quality"])),
        "diversity": float(np.mean(details["diversity"])),
        "controlability": float(np.mean(details["controlability"])),
        # fraction of levels that fully pass each criterion (the benchmark's own scores)
        "quality_pass": float(q_score),
        "diversity_pass": float(d_score),
        "controlability_pass": float(c_score),
        "per_level": {"quality": to_python(details["quality"]),
                      "diversity": to_python(details["diversity"]),
                      "controlability": to_python(details["controlability"])},
        "mean_time": float(np.mean([r["time"] for r in runs])),
        "mean_iterations": float(np.mean([r["iterations"] for r in runs])),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generators on quality, diversity and controlability. Extra --key value pairs are passed to every generator's reset.")
    parser.add_argument("outputfolder", type=str, help="folder for the per-run results and summaries")
    parser.add_argument("-p", "--problem", type=str, default="smb-small-v0")
    parser.add_argument("-g", "--generators", type=str, nargs="+", default=["genetic", "random_search"], help="generator file names in generators/")
    parser.add_argument("-n", "--runs", type=int, default=20, help="runs (= levels in the evaluated set) per generator")
    parser.add_argument("-s", "--steps", type=int, default=400, help="max updates per run")
    parser.add_argument("--seed", type=int, default=1, help="seed of the first run (seed 0 is ignored by the generators' reset)")
    parser.add_argument("-e", "--early_stop", action="store_true", help="stop a run once best() reaches 1")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count())
    parser.add_argument("--single", type=str, nargs="*", default=[], help="generators evaluated with ONE run whose result serves all N targets (e.g. map_elites); they must accept workers= in reset")
    args, additional = parser.parse_known_args()
    gen_args = convert2Dic(additional)

    os.makedirs(args.outputfolder, exist_ok=True)
    summary = {}
    for name in args.generators:
        jobs = [(args.problem, name, args.seed + i, args.steps, args.early_stop, gen_args,
                 os.path.join(args.outputfolder, name)) for i in range(args.runs)]
        folder = os.path.join(args.outputfolder, name)
        os.makedirs(folder, exist_ok=True)
        if name in args.single:
            print(f"{name}: one run of {args.steps} steps serving {args.runs} targets", flush=True)
            generator, runs = run_single(args.problem, name, [args.seed + i for i in range(args.runs)],
                                         args.steps, args.workers, gen_args)
            if hasattr(generator, "save"):
                generator.save(os.path.join(folder, "archive"))
        else:
            print(f"{name}: {args.runs} runs, up to {args.steps} steps each", flush=True)
            runs = []
            with Pool(min(args.workers, args.runs)) as pool:
                for r in pool.imap_unordered(run_once, jobs):
                    runs.append(r)
                    last = r["history"][-1]
                    print(f"  [{len(runs)}/{args.runs}] seed {r['seed']}: best {last['best']:.3f} after "
                          f"{r['iterations']} steps, {r['time']:.0f}s", flush=True)
            runs.sort(key=lambda r: r["seed"])

        for r in runs:
            with open(os.path.join(folder, f"run_{r['seed']}.json"), "w") as f:
                json.dump(r, f)
        env = pcg_benchmark.make(args.problem)
        for r in runs[:5]:
            env.render(r["content"]).save(os.path.join(folder, f"level_{r['seed']}.png"))

        summary[name] = score_set(args.problem, runs)
        s = summary[name]
        print(f"  quality {s['quality']:.3f} ({s['quality_pass']:.0%} pass)  "
              f"diversity {s['diversity']:.3f} ({s['diversity_pass']:.0%})  "
              f"controlability {s['controlability']:.3f} ({s['controlability_pass']:.0%})  "
              f"{s['mean_time']:.1f}s/run")

    # merge with earlier summaries so generators can be evaluated in separate calls
    summary_path = os.path.join(args.outputfolder, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = {**json.load(f), **summary}
    with open(summary_path, "w") as f:
        json.dump({**summary, "_config": {"problem": args.problem, "runs": args.runs, "steps": args.steps,
                                          "seed": args.seed, "early_stop": args.early_stop, "gen_args": gen_args}}, f, indent=2)
    print(f"saved {summary_path}")
