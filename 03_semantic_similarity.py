import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def read_table(path):
    path = Path(path)
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def write_table(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=index, encoding="utf-8-sig")
    else:
        frame.to_excel(path, index=index)


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def pairwise_similarity(input_file, output_file, id_column, text_column, model_name):
    frame = read_table(input_file)
    missing = {id_column, text_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    ids = clean_text(frame[id_column])
    texts = clean_text(frame[text_column])
    valid = (ids != "") & (texts != "")
    ids = ids[valid]
    texts = texts[valid]
    if ids.duplicated().any():
        raise ValueError("IDs must be unique")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts.tolist(), normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(embeddings) @ np.asarray(embeddings).T
    result = pd.DataFrame(matrix, index=ids.tolist(), columns=ids.tolist()).round(3)
    write_table(result, output_file, index=True)
    return result


def matched_similarity(
    input_file,
    output_file,
    original_id_column,
    original_text_column,
    final_id_column,
    final_text_column,
    match_column,
    model_name,
):
    frame = read_table(input_file)
    required = {
        original_id_column,
        original_text_column,
        final_id_column,
        final_text_column,
        match_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    final_map = (
        frame[[final_id_column, final_text_column]]
        .dropna(subset=[final_id_column])
        .drop_duplicates(final_id_column)
        .set_index(final_id_column)[final_text_column]
        .to_dict()
    )
    matched_text = frame[match_column].map(final_map)
    if matched_text.isna().any():
        raise ValueError("Some matched final IDs were not found")
    original_text = clean_text(frame[original_text_column])
    final_text = clean_text(matched_text)
    if (original_text == "").any() or (final_text == "").any():
        raise ValueError("Text values cannot be blank")
    model = SentenceTransformer(model_name)
    original_embeddings = np.asarray(
        model.encode(original_text.tolist(), normalize_embeddings=True, show_progress_bar=True)
    )
    final_embeddings = np.asarray(
        model.encode(final_text.tolist(), normalize_embeddings=True, show_progress_bar=True)
    )
    result = pd.DataFrame(
        {
            "original_id": frame[original_id_column],
            "final_id": frame[match_column],
            "semantic_similarity": np.sum(original_embeddings * final_embeddings, axis=1).round(3),
        }
    )
    write_table(result, output_file)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    pairwise = subparsers.add_parser("pairwise")
    pairwise.add_argument("input_file")
    pairwise.add_argument("output_file")
    pairwise.add_argument("--id-column", default="original_id")
    pairwise.add_argument("--text-column", default="original_text")
    matched = subparsers.add_parser("matched")
    matched.add_argument("input_file")
    matched.add_argument("output_file")
    matched.add_argument("--original-id-column", default="original_id")
    matched.add_argument("--original-text-column", default="original_text")
    matched.add_argument("--final-id-column", default="final_id")
    matched.add_argument("--final-text-column", default="final_text")
    matched.add_argument("--match-column", default="contained_in")
    for subparser in [pairwise, matched]:
        subparser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "pairwise":
        pairwise_similarity(args.input_file, args.output_file, args.id_column, args.text_column, args.model)
    else:
        matched_similarity(
            args.input_file,
            args.output_file,
            args.original_id_column,
            args.original_text_column,
            args.final_id_column,
            args.final_text_column,
            args.match_column,
            args.model,
        )


if __name__ == "__main__":
    main()
