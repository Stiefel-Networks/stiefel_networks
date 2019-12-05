import os
import json
import gc

from mnist_F_SVD.run import OUT_PATH

def main():
    run_files = [f for f in os.listdir(OUT_PATH) if os.path.isfile(os.path.join(OUT_PATH, f))]
    run_files.sort()

    out_directory = os.path.join(OUT_PATH, "no_batch_sv")
    os.makedirs(out_directory, exist_ok=True)

    for run_filname in run_files:
        run_data = None
        with open(os.path.join(OUT_PATH, run_filname)) as file:
            # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
            # do its thing in advance of loading the next one.
            gc.collect()

            print("Loading {}...".format(run_filname))
            run_data = json.loads(file.read())

            print("Loaded:\n  {}".format(run_data["hyperparameters"]))

            for batch_progress in run_data["batches_progress"]:
                if "singular_values" in batch_progress:
                    del batch_progress["singular_values"]

        with open(os.path.join(out_directory, run_filname), "w") as file:
            file.write(json.dumps(run_data, indent=2))


if __name__ == "__main__":
    main()