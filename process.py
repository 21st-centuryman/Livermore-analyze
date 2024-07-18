import polars as pl
from tqdm import tqdm, trange
import os
import concurrent.futures


def process_file(file, path, seq_length):
    if file.endswith(".csv"):
        df = pl.read_csv(f"{path}/{file}").drop("TIMESTAMP")
        tape = pl.DataFrame(
            {"TAPE": [x for pair in zip(df["OPEN"], df["CLOSE"]) for x in pair]}
        ).to_numpy()
        dh = pl.DataFrame()
        for i in trange(len(tape) - seq_length, leave=False):
            seq_vals = tape[i : i + seq_length + 1].tolist()
            new_dh = pl.DataFrame(seq_vals)
            dh = pl.concat([dh, new_dh])
        dh.write_csv(f"process_data/{seq_length}/{file}")


def main(path, seq_length):
    files = os.listdir(path)
    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(
        total=len(files)
    ) as pbar:
        futures = {
            executor.submit(process_file, file, path, seq_length): file
            for file in files
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            pbar.update(1)


if __name__ == "__main__":
    path = "../data"
    seq_length = 30
    os.makedirs("process_data", exist_ok=True)
    os.makedirs(f"process_data/{seq_length}", exist_ok=True)
    main(path, seq_length)
