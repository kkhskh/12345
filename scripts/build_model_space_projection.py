#!/usr/bin/env python3
"""Build preliminary 3D model-space projections from the T0 atlas.

This script reads the model atlas CSV, validates it against the feature schema,
encodes features (boolean -> 0/1, numeric -> float, categorical -> one-hot),
builds a feature matrix for a chosen projection config, and reduces it to 3D
using PCA, t-SNE (both via scikit-learn), or UMAP (via umap-learn).

All heavy dependencies are optional. The script degrades gracefully:
  - No numpy/scikit-learn: it still writes the encoded feature matrix and warns
    that no dimensionality reduction was performed.
  - No umap-learn: the ``umap`` method is skipped with a warning.
  - No matplotlib: the scatter-plot step is skipped with a warning.

It performs no network access.

Example
-------
    python scripts/build_model_space_projection.py \
        --input data/model_atlas/top_100_models_template.csv \
        --schema data/model_atlas/model_feature_schema_v0.json \
        --projection mechanistic_access \
        --method pca \
        --output artifacts/model_space/model_space_coordinates_v0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Optional

# Columns that are always treated as categorical (one-hot) rather than numeric.
CATEGORICAL_COLUMNS = {
    "access_type",
    "mechanistic_access",
    "family",
    "license",
    "verification_status",
    "provider",
    "atlas_role",
}

# Columns that identify a row and should never be encoded as features.
IDENTIFIER_COLUMNS = {"model_id", "canonical_name", "source_notes", "notes_project_status"}

# Truthy/falsy string tokens recognized for boolean-like columns.
TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
FALSE_TOKENS = {"0", "false", "no", "n", "f"}


def eprint(*args: Any) -> None:
    """Print to stderr and flush (used for warnings and errors)."""

    print(*args, file=sys.stderr, flush=True)


def load_schema(schema_path: str) -> dict[str, Any]:
    """Load the feature-schema JSON, raising a clear error on failure."""

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    try:
        with open(schema_path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema file {schema_path} is not valid JSON: {exc}") from exc
    if "feature_blocks" not in schema or "projection_configs" not in schema:
        raise ValueError(
            f"Schema {schema_path} must contain 'feature_blocks' and 'projection_configs'."
        )
    return schema


def load_rows(input_path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Load CSV rows as dictionaries, returning (header, rows)."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV {input_path} has no header row.")
        header = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Input CSV {input_path} has no data rows.")
    return header, rows


def validate_required_columns(header: list[str], schema: dict[str, Any]) -> list[str]:
    """Return the list of required columns missing from the header."""

    required = schema.get("required_columns", [])
    return [column for column in required if column not in header]


def resolve_projection_columns(
    schema: dict[str, Any], projection: str, header: list[str]
) -> tuple[list[str], list[str], bool]:
    """Resolve the feature columns for a projection config.

    Returns (present_columns, missing_columns, include_closed_models).
    """

    projection_configs = schema.get("projection_configs", {})
    if projection not in projection_configs:
        available = ", ".join(sorted(projection_configs)) or "(none)"
        raise ValueError(
            f"Unknown projection '{projection}'. Available projections: {available}."
        )
    config = projection_configs[projection]
    feature_blocks = schema.get("feature_blocks", {})
    wanted: list[str] = []
    for block_name in config.get("feature_blocks", []):
        for column in feature_blocks.get(block_name, []):
            if column in IDENTIFIER_COLUMNS:
                continue
            if column not in wanted:
                wanted.append(column)
    present = [column for column in wanted if column in header]
    missing = [column for column in wanted if column not in header]
    include_closed = bool(config.get("include_closed_models", True))
    return present, missing, include_closed


def _is_boolean_column(values: list[str]) -> bool:
    """Return True if all non-empty values are boolean-like tokens."""

    non_empty = [v.strip().lower() for v in values if v.strip() != ""]
    if not non_empty:
        return False
    return all(v in TRUE_TOKENS or v in FALSE_TOKENS for v in non_empty)


def _is_numeric_column(values: list[str]) -> bool:
    """Return True if all non-empty values parse as floats."""

    non_empty = [v.strip() for v in values if v.strip() != ""]
    if not non_empty:
        return False
    for value in non_empty:
        try:
            float(value)
        except ValueError:
            return False
    return True


def _to_bool(value: str) -> Optional[float]:
    token = value.strip().lower()
    if token in TRUE_TOKENS:
        return 1.0
    if token in FALSE_TOKENS:
        return 0.0
    return None


def encode_features(
    rows: list[dict[str, str]], columns: list[str]
) -> tuple[list[str], list[list[float]], list[str], dict[str, str]]:
    """Encode selected columns into a numeric feature matrix.

    Returns (feature_names, matrix, skipped_columns, column_kinds).
    Missing numeric values are filled with the column mean; missing categorical
    values become an explicit ``__missing__`` one-hot level.
    """

    feature_names: list[str] = []
    column_kinds: dict[str, str] = {}
    skipped: list[str] = []
    # Column-major encoded blocks, later stitched into row-major matrix.
    encoded_columns: list[list[float]] = []

    for column in columns:
        raw_values = [row.get(column, "") for row in rows]
        if all(v.strip() == "" for v in raw_values):
            skipped.append(column)
            column_kinds[column] = "empty"
            continue

        if column in CATEGORICAL_COLUMNS or not (
            _is_boolean_column(raw_values) or _is_numeric_column(raw_values)
        ):
            column_kinds[column] = "categorical"
            levels = sorted({(v.strip() or "__missing__") for v in raw_values})
            for level in levels:
                feature_names.append(f"{column}={level}")
                encoded_columns.append(
                    [1.0 if (v.strip() or "__missing__") == level else 0.0 for v in raw_values]
                )
            continue

        if _is_boolean_column(raw_values):
            column_kinds[column] = "boolean"
            parsed = [_to_bool(v) for v in raw_values]
        else:
            column_kinds[column] = "numeric"
            parsed = []
            for v in raw_values:
                token = v.strip()
                parsed.append(float(token) if token != "" else None)

        known = [p for p in parsed if p is not None]
        mean = sum(known) / len(known) if known else 0.0
        filled = [p if p is not None else mean for p in parsed]
        feature_names.append(column)
        encoded_columns.append(filled)

    n_rows = len(rows)
    matrix = [[encoded_columns[c][r] for c in range(len(encoded_columns))] for r in range(n_rows)]
    return feature_names, matrix, skipped, column_kinds


def standardize(matrix: list[list[float]]) -> list[list[float]]:
    """Standardize columns to zero mean and unit variance (pure Python)."""

    if not matrix or not matrix[0]:
        return matrix
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = [sum(matrix[r][c] for r in range(n_rows)) / n_rows for c in range(n_cols)]
    stds = []
    for c in range(n_cols):
        var = sum((matrix[r][c] - means[c]) ** 2 for r in range(n_rows)) / n_rows
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    return [
        [(matrix[r][c] - means[c]) / stds[c] for c in range(n_cols)]
        for r in range(n_rows)
    ]


def reduce_dimensionality(
    matrix: list[list[float]], method: str, n_components: int = 3
) -> tuple[Optional[list[list[float]]], list[str]]:
    """Reduce the feature matrix to n_components dimensions.

    Returns (coordinates or None, warnings). Coordinates is None when the
    requested method's dependencies are unavailable or the matrix is too small.
    """

    warnings: list[str] = []
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if n_rows == 0 or n_cols == 0:
        return None, ["No features available for dimensionality reduction."]

    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None, [
            "numpy is not installed; skipping dimensionality reduction. "
            "Encoded feature matrix will still be written."
        ]

    data = np.asarray(matrix, dtype=float)
    effective_components = min(n_components, n_rows, n_cols)
    if effective_components < 1:
        return None, ["Not enough data to compute any component."]
    if effective_components < n_components:
        warnings.append(
            f"Requested {n_components} components but only {effective_components} "
            f"are possible for {n_rows} rows and {n_cols} features."
        )

    method = method.lower()
    coords: Optional[Any] = None

    if method == "pca":
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except ImportError:
            return None, warnings + [
                "scikit-learn is not installed; skipping PCA."
            ]
        coords = PCA(n_components=effective_components, random_state=0).fit_transform(data)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore
        except ImportError:
            return None, warnings + [
                "scikit-learn is not installed; skipping t-SNE."
            ]
        perplexity = max(2.0, min(30.0, (n_rows - 1) / 3.0))
        coords = TSNE(
            n_components=effective_components,
            random_state=0,
            perplexity=perplexity,
            init="random",
        ).fit_transform(data)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError:
            return None, warnings + [
                "umap-learn is not installed; skipping UMAP. "
                "Install umap-learn or use --method pca."
            ]
        n_neighbors = max(2, min(15, n_rows - 1))
        coords = umap.UMAP(
            n_components=effective_components, random_state=0, n_neighbors=n_neighbors
        ).fit_transform(data)
    else:
        return None, warnings + [f"Unknown method '{method}'. Use pca, tsne, or umap."]

    coords_list = [[float(v) for v in row] for row in coords]
    # Pad to exactly n_components columns so the output schema is stable.
    for row in coords_list:
        while len(row) < n_components:
            row.append(0.0)
    return coords_list, warnings


def write_coordinates(
    output_path: str,
    rows: list[dict[str, str]],
    coords: Optional[list[list[float]]],
    feature_matrix: list[list[float]],
    feature_names: list[str],
    projection: str,
    method: str,
) -> None:
    """Write coordinates (or raw features if reduction was skipped) to CSV."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if coords is not None:
            writer.writerow(
                ["model_id", "canonical_name", "family", "access_type",
                 "mechanistic_access", "projection", "method", "dim1", "dim2", "dim3"]
            )
            for row, coord in zip(rows, coords):
                writer.writerow(
                    [
                        row.get("model_id", ""),
                        row.get("canonical_name", ""),
                        row.get("family", ""),
                        row.get("access_type", ""),
                        row.get("mechanistic_access", ""),
                        projection,
                        method,
                        coord[0],
                        coord[1],
                        coord[2],
                    ]
                )
        else:
            # Fall back to writing the encoded feature matrix.
            writer.writerow(["model_id", "canonical_name", "projection"] + feature_names)
            for row, features in zip(rows, feature_matrix):
                writer.writerow(
                    [row.get("model_id", ""), row.get("canonical_name", ""), projection]
                    + features
                )


def write_scatter_plot(
    coords: Optional[list[list[float]]],
    rows: list[dict[str, str]],
    projection: str,
    method: str,
    figure_path: str,
) -> list[str]:
    """Write a 3D scatter plot if matplotlib is available."""

    if coords is None:
        return ["No coordinates available; skipping scatter plot."]
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    except ImportError:
        return ["matplotlib is not installed; skipping scatter plot."]

    os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)
    families = [row.get("family", "unknown") or "unknown" for row in rows]
    unique_families = sorted(set(families))
    family_to_index = {family: idx for idx, family in enumerate(unique_families)}
    colors = [family_to_index[family] for family in families]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap="tab20", s=40, depthshade=True)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    ax.set_title(f"Model space projection: {projection} ({method})")

    # Build a compact legend keyed by family index.
    handles = []
    for family in unique_families:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                color=scatter.cmap(scatter.norm(family_to_index[family])),
                label=family,
            )
        )
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [f"Wrote scatter plot to {figure_path}."]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preliminary 3D model-space projections from the T0 atlas."
    )
    parser.add_argument(
        "--input",
        default="data/model_atlas/top_100_models_template.csv",
        help="Path to the model atlas CSV.",
    )
    parser.add_argument(
        "--schema",
        default="data/model_atlas/model_feature_schema_v0.json",
        help="Path to the feature schema JSON.",
    )
    parser.add_argument(
        "--projection",
        default="mechanistic_access",
        help="Projection config name from the schema's projection_configs.",
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=["pca", "tsne", "umap"],
        help="Dimensionality-reduction method.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/model_space/model_space_coordinates_v0.csv",
        help="Output CSV path for coordinates.",
    )
    parser.add_argument(
        "--figure",
        default="artifacts/figures/model_space_projection_v0.png",
        help="Output PNG path for the 3D scatter plot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    all_warnings: list[str] = []

    try:
        schema = load_schema(args.schema)
        header, rows = load_rows(args.input)
    except (FileNotFoundError, ValueError) as exc:
        eprint(f"ERROR: {exc}")
        return 1

    missing_required = validate_required_columns(header, schema)
    if missing_required:
        eprint(
            "ERROR: input CSV is missing required columns: "
            + ", ".join(missing_required)
        )
        return 1

    try:
        columns, missing_projection, include_closed = resolve_projection_columns(
            schema, args.projection, header
        )
    except ValueError as exc:
        eprint(f"ERROR: {exc}")
        return 1

    total_rows = len(rows)
    if not include_closed:
        closed_values = {"closed_api_ui", "closed_system_wrapper", "media_api"}
        rows = [r for r in rows if r.get("access_type", "").strip() not in closed_values]
        if not rows:
            eprint(
                f"ERROR: projection '{args.projection}' excludes closed models, "
                "but no rows remain after filtering."
            )
            return 1

    if not columns:
        eprint(
            f"ERROR: no schema feature columns for projection '{args.projection}' "
            "are present in the input CSV."
        )
        return 1

    feature_names, matrix, skipped_empty, column_kinds = encode_features(rows, columns)
    matrix = standardize(matrix)
    coords, reduce_warnings = reduce_dimensionality(matrix, args.method)
    all_warnings.extend(reduce_warnings)

    try:
        write_coordinates(
            args.output, rows, coords, matrix, feature_names, args.projection, args.method
        )
    except OSError as exc:
        eprint(f"ERROR: could not write output CSV: {exc}")
        return 1

    plot_messages = write_scatter_plot(
        coords, rows, args.projection, args.method, args.figure
    )
    all_warnings.extend(plot_messages)

    print("=" * 70, flush=True)
    print("build_model_space_projection summary", flush=True)
    print("=" * 70, flush=True)
    print(f"input CSV:            {args.input}", flush=True)
    print(f"schema:               {args.schema}", flush=True)
    print(f"projection:           {args.projection}", flush=True)
    print(f"method:               {args.method}", flush=True)
    print(f"rows in file:         {total_rows}", flush=True)
    print(f"rows included:        {len(rows)} (closed models "
          f"{'included' if include_closed else 'excluded'})", flush=True)
    print(f"feature columns used: {len([c for c in columns if c not in skipped_empty])}", flush=True)
    print(f"encoded dimensions:   {len(feature_names)}", flush=True)
    if missing_projection:
        print(
            "schema columns not in CSV (skipped): " + ", ".join(missing_projection),
            flush=True,
        )
    if skipped_empty:
        print("present-but-empty columns (skipped): " + ", ".join(skipped_empty), flush=True)
    categorical = [c for c, k in column_kinds.items() if k == "categorical"]
    if categorical:
        print("one-hot encoded columns: " + ", ".join(categorical), flush=True)
    if coords is not None:
        print(f"coordinates written:  {args.output}", flush=True)
    else:
        print(f"encoded features written (no reduction): {args.output}", flush=True)
    if all_warnings:
        print("-" * 70, flush=True)
        print("notes / warnings:", flush=True)
        for warning in all_warnings:
            print(f"  - {warning}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
