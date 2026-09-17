"""Print a matched MOSI reconstruction-loss control plan; never launches jobs.

Keep the reconstruction architecture active in both conditions, varying only
its auxiliary loss coefficient. Separate output roots prevent coefficient
conditions from sharing result filenames.
"""
import shlex


def commands():
    for coefficient, condition in [(0.0, "loss0"), (0.5, "loss05")]:
        for protocol in ("IMM", "FMM"):
            for regime in ("U-lo", "T-frag"):
                for seed in range(5):
                    yield [
                        "python", "-m", "parm.train", "--dataset", "MOSI",
                        "--method", "recon", "--text-encoder", "bert",
                        "--protocol", protocol, "--train-regime", regime,
                        "--seed", str(seed), "--lambda-recon", str(coefficient),
                        "--out", f"results_controls/{condition}",
                    ]


if __name__ == "__main__":
    for command in commands():
        print(shlex.join(command))
