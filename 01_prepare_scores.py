import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CRITERIA = ["C1", "C2", "C3", "C4", "C5"]
REQUIRED_COLUMNS = ["idea_id", "research_idea", *CRITERIA]
VALID_SCORES = {0.0, 0.5, 1.0}


def read_table(path):
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def normalize_score(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    score = float(str(value).strip().replace("·", "."))
    if score not in VALID_SCORES:
        raise ValueError(f"Invalid score: {value}")
    return score


def read_metadata(path):
    if path is None:
        return None
    metadata = read_table(path)
    if "filename" not in metadata.columns:
        raise ValueError("Metadata must contain filename")
    if metadata["filename"].duplicated().any():
        raise ValueError("Metadata contains duplicate filenames")
    return metadata.set_index("filename")


def prepare_scores(input_dir, output_file, metadata_file=None):
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".xlsx", ".xls", ".csv"} and not path.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError("No score files found")
    metadata = read_metadata(Path(metadata_file) if metadata_file else None)
    parts = []
    for position, path in enumerate(files, start=1):
        frame = read_table(path)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Input file {position} is missing columns: {sorted(missing)}")
        frame = frame[REQUIRED_COLUMNS].copy()
        for criterion in CRITERIA:
            frame[criterion] = frame[criterion].map(normalize_score)
        long_frame = frame.melt(
            id_vars=["idea_id", "research_idea"],
            value_vars=CRITERIA,
            var_name="criterion",
            value_name="score",
        )
        long_frame["expert_id"] = f"E{position:03d}"
        long_frame["WHO_region"] = np.nan
        long_frame["income_group"] = np.nan
        if metadata is not None:
            if path.name not in metadata.index:
                raise ValueError(f"Input file {position} is absent from metadata")
            row = metadata.loc[path.name]
            region_column = next((name for name in ["WHO_region", "who_region"] if name in metadata.columns), None)
            income_column = next((name for name in ["income_group", "Income_group"] if name in metadata.columns), None)
            if region_column:
                long_frame["WHO_region"] = row[region_column]
            if income_column:
                long_frame["income_group"] = row[income_column]
        parts.append(long_frame)
    scores = pd.concat(parts, ignore_index=True)
    scores = scores[
        ["expert_id", "idea_id", "research_idea", "criterion", "score", "WHO_region", "income_group"]
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_file, index=False, encoding="utf-8-sig")
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    parser.add_argument("--metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    scores = prepare_scores(args.input_dir, args.output_file, args.metadata)
    print({"experts": scores["expert_id"].nunique(), "ideas": scores["idea_id"].nunique(), "rows": len(scores)})


if __name__ == "__main__":
    main()
