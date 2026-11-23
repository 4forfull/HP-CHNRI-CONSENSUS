import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


CRITERIA = ["C1", "C2", "C3", "C4", "C5"]
SCORE_CATEGORIES = [1.0, 0.5, 0.0]
REGION_COLUMNS = ["WHO_region", "who_region", "region"]
INCOME_COLUMNS = ["income_group", "Income_group"]


def validate_scores(frame):
    required = {"expert_id", "idea_id", "research_idea", "criterion", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    invalid = frame.loc[frame["score"].notna() & ~frame["score"].isin(SCORE_CATEGORIES), "score"]
    if not invalid.empty:
        raise ValueError(f"Invalid scores: {sorted(invalid.unique())}")
    invalid_criteria = set(frame["criterion"].dropna().unique()) - set(CRITERIA)
    if invalid_criteria:
        raise ValueError(f"Invalid criteria: {sorted(invalid_criteria)}")
    return frame


def detect_column(frame, candidates):
    return next((column for column in candidates if column in frame.columns), None)


def entropy_from_probabilities(probabilities):
    positive = probabilities[probabilities > 0]
    return float(np.exp(np.sum(positive * np.log(positive))))


def entropy_agreement(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    counts = np.array([(values == category).sum() for category in SCORE_CATEGORIES], dtype=float)
    return entropy_from_probabilities(counts / counts.sum())


def add_region_weights(frame, region_column):
    mapping = frame[["expert_id", region_column]].drop_duplicates()
    if mapping["expert_id"].duplicated().any():
        raise ValueError("Each expert must map to one region")
    if mapping[region_column].isna().any():
        raise ValueError("Region values cannot be missing in weighted analysis")
    counts = mapping[region_column].value_counts()
    output = frame.copy()
    output["analysis_weight"] = output[region_column].map(counts.rdiv(1.0))
    return output


def weighted_entropy(group):
    group = group.dropna(subset=["score"])
    total = group["analysis_weight"].sum()
    if group.empty or total == 0:
        return np.nan
    probabilities = np.array(
        [group.loc[group["score"] == category, "analysis_weight"].sum() / total for category in SCORE_CATEGORIES]
    )
    return entropy_from_probabilities(probabilities)


def calculate_components(frame, weighted=False, region_column=None):
    data = add_region_weights(frame, region_column) if weighted else frame.copy()
    nonblank = data.dropna(subset=["score"]).copy()
    keys = ["idea_id", "research_idea", "criterion"]
    if weighted:
        nonblank["weighted_score"] = nonblank["score"] * nonblank["analysis_weight"]
        css = nonblank.groupby(keys, as_index=False).agg(
            weighted_score=("weighted_score", "sum"),
            total_weight=("analysis_weight", "sum"),
        )
        css["CSS"] = css["weighted_score"] / css["total_weight"]
        css = css[keys + ["CSS"]]
        aea = nonblank.groupby(["idea_id", "criterion"], group_keys=False).apply(weighted_entropy)
    else:
        css = nonblank.groupby(keys, as_index=False)["score"].mean().rename(columns={"score": "CSS"})
        aea = nonblank.groupby(["idea_id", "criterion"])["score"].apply(entropy_agreement)
    aea = aea.reset_index(name="AEA_component")
    return css, aea


def calculate_scores(frame, weighted=False, region_column=None):
    css, aea = calculate_components(frame, weighted, region_column)
    wide = css.pivot(index=["idea_id", "research_idea"], columns="criterion", values="CSS").reset_index()
    for criterion in CRITERIA:
        if criterion not in wide.columns:
            wide[criterion] = np.nan
    wide = wide[["idea_id", "research_idea", *CRITERIA]]
    wide["RPS"] = wide[CRITERIA].mean(axis=1)
    overall_aea = aea.groupby("idea_id", as_index=False)["AEA_component"].mean().rename(
        columns={"AEA_component": "AEA_e"}
    )
    scores = wide.merge(overall_aea, on="idea_id", how="left")
    scores["rank"] = scores["RPS"].rank(method="min", ascending=False).astype("Int64")
    return scores.sort_values(["rank", "idea_id"]), css, aea


def bootstrap_sample(frame, rng, weighted=False, region_column=None):
    expert_columns = ["expert_id"] + ([region_column] if weighted else [])
    experts = frame[expert_columns].drop_duplicates()
    if experts["expert_id"].duplicated().any():
        raise ValueError("Each expert must map to one region")
    sampled = []
    draw = 0
    groups = experts.groupby(region_column, dropna=False) if weighted else [(None, experts)]
    for _, group in groups:
        selected = rng.choice(group["expert_id"].to_numpy(), size=len(group), replace=True)
        for expert_id in selected:
            draw += 1
            part = frame.loc[frame["expert_id"] == expert_id].copy()
            part["expert_id"] = f"B{draw:04d}"
            sampled.append(part)
    return pd.concat(sampled, ignore_index=True)


def run_bootstrap(frame, iterations, seed, weighted=False, region_column=None):
    rng = np.random.default_rng(seed)
    results = []
    for bootstrap_id in range(1, iterations + 1):
        sample = bootstrap_sample(frame, rng, weighted, region_column)
        scores, _, _ = calculate_scores(sample, weighted, region_column)
        scores["bootstrap_id"] = bootstrap_id
        results.append(scores)
    return pd.concat(results, ignore_index=True)


def add_confidence_intervals(original, bootstrap):
    output = original.copy()
    for metric in [*CRITERIA, "RPS", "AEA_e"]:
        interval = bootstrap.groupby("idea_id")[metric].quantile([0.025, 0.975]).unstack()
        interval.columns = [f"{metric}_lower", f"{metric}_upper"]
        output = output.merge(interval.reset_index(), on="idea_id", how="left")
    return output


def rank_probabilities(bootstrap):
    ranks = bootstrap[["bootstrap_id", "idea_id", "rank"]].copy()
    total = ranks["bootstrap_id"].nunique()
    distribution = ranks.groupby(["idea_id", "rank"]).size().reset_index(name="count")
    distribution["probability"] = distribution["count"] / total
    top10 = ranks.assign(in_top10=ranks["rank"] <= 10).groupby("idea_id", as_index=False)["in_top10"].mean()
    top10 = top10.rename(columns={"in_top10": "top10_probability"})
    return distribution, top10


def icc_3_1(x, y):
    values = pd.DataFrame({"x": x, "y": y}).dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    grand = values.mean()
    target_means = values.mean(axis=1, keepdims=True)
    rater_means = values.mean(axis=0, keepdims=True)
    ms_targets = 2 * np.sum((target_means - grand) ** 2) / (len(values) - 1)
    ms_error = np.sum((values - target_means - rater_means + grand) ** 2) / (len(values) - 1)
    denominator = ms_targets + ms_error
    return np.nan if denominator == 0 else float((ms_targets - ms_error) / denominator)


def correlation_row(name, score, agreement):
    pair = pd.DataFrame({"score": score, "agreement": agreement}).dropna()
    if len(pair) < 3:
        return {"analysis": name, "n_ideas": len(pair), "spearman_rho": np.nan, "p_value": np.nan, "icc_3_1": np.nan}
    rho, p_value = spearmanr(pair["score"], pair["agreement"])
    return {
        "analysis": name,
        "n_ideas": len(pair),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "icc_3_1": icc_3_1(pair["score"], pair["agreement"]),
    }


def correlation_summary(scores, css, aea):
    rows = [correlation_row("Overall", scores["RPS"], scores["AEA_e"])]
    criterion_data = css.merge(aea, on=["idea_id", "criterion"], how="inner")
    for criterion in CRITERIA:
        subset = criterion_data.loc[criterion_data["criterion"] == criterion]
        rows.append(correlation_row(criterion, subset["CSS"], subset["AEA_component"]))
    return pd.DataFrame(rows)


def criterion_ranks(scores):
    parts = []
    for criterion in CRITERIA:
        part = scores[["idea_id", "research_idea", criterion, f"{criterion}_lower", f"{criterion}_upper"]].copy()
        part = part.rename(
            columns={criterion: "CSS", f"{criterion}_lower": "CSS_lower", f"{criterion}_upper": "CSS_upper"}
        )
        part["criterion"] = criterion
        part["rank"] = part["CSS"].rank(method="min", ascending=False).astype("Int64")
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["criterion", "rank", "idea_id"])


def safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return label or "group"


def run_analysis(frame, output_dir, label, iterations, seed, weighted=False, region_column=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, css, aea = calculate_scores(frame, weighted, region_column)
    bootstrap = run_bootstrap(frame, iterations, seed, weighted, region_column)
    scores_ci = add_confidence_intervals(scores, bootstrap)
    rank_distribution, top10 = rank_probabilities(bootstrap)
    scores_ci = scores_ci.merge(top10, on="idea_id", how="left")
    scores_ci.insert(0, "analysis", label)
    scores_ci.to_csv(output_dir / "scores_with_ci.csv", index=False, encoding="utf-8-sig")
    rank_distribution.to_csv(output_dir / "rank_probability.csv", index=False, encoding="utf-8-sig")
    criterion_ranks(scores_ci).to_csv(output_dir / "criterion_ranks.csv", index=False, encoding="utf-8-sig")
    correlation_summary(scores, css, aea).to_csv(output_dir / "correlation_summary.csv", index=False, encoding="utf-8-sig")


def cluster_orders(frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = []
    linkages = []
    for analysis in ["Overall", *CRITERIA]:
        selected = frame if analysis == "Overall" else frame.loc[frame["criterion"] == analysis]
        matrix = selected.pivot_table(index="expert_id", columns="idea_id", values="score", aggfunc="mean")
        if len(matrix) < 2:
            continue
        filled = matrix.fillna(matrix.mean(axis=0)).fillna(0.5)
        tree = linkage(pdist(filled.to_numpy(), metric="euclidean"), method="average", optimal_ordering=True)
        ordered_ids = filled.index[leaves_list(tree)]
        orders.extend({"analysis": analysis, "order": position, "expert_id": expert_id} for position, expert_id in enumerate(ordered_ids, start=1))
        linkages.extend(
            {
                "analysis": analysis,
                "node_left": int(row[0]),
                "node_right": int(row[1]),
                "distance": float(row[2]),
                "sample_count": int(row[3]),
            }
            for row in tree
        )
    pd.DataFrame(orders).to_csv(output_dir / "cluster_order.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(linkages).to_csv(output_dir / "cluster_linkage.csv", index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--skip-weighted", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = validate_scores(pd.read_csv(args.input_file))
    output_dir = Path(args.output_dir)
    run_analysis(frame, output_dir / "global", "Global", args.bootstrap, args.seed)
    region_column = detect_column(frame, REGION_COLUMNS)
    if not args.skip_weighted and region_column:
        run_analysis(
            frame,
            output_dir / "who_region_weighted",
            "WHO region weighted",
            args.bootstrap,
            args.seed + 1,
            weighted=True,
            region_column=region_column,
        )
    income_column = detect_column(frame, INCOME_COLUMNS)
    if not args.skip_subgroups and income_column:
        groups = [value for value in frame[income_column].dropna().unique() if str(value).strip()]
        for position, group in enumerate(groups, start=1):
            subset = frame.loc[frame[income_column] == group].copy()
            run_analysis(
                subset,
                output_dir / f"income_{safe_label(group)}",
                f"Income group {group}",
                args.bootstrap,
                args.seed + 100 + position,
            )
    if not args.skip_clustering:
        cluster_orders(frame, output_dir / "clustering")


if __name__ == "__main__":
    main()
