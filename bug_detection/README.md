# Paper materials: LLM bug detection (documentation conditions)

This directory contains the **dataset**, **bug-detection pipeline script**, and **frozen DeepSeek model outputs** used for the empirical study on LLM-based bug detection under four documentation conditions.

---

## Contents

| File | Description |
|------|-------------|
| `strengthened_comments_full.json` | Input dataset: one entry per bug-fixing change, with focal method in **prefix (buggy)** form, original/updated comments, and strengthened comments where applicable. |
| `llm_analyzed_buggy_method_sourcecode_deepseek.json` | Model outputs for **M1** (code only). |
| `llm_analyzed_prefix_method_plus_comment_deepseek.json` | Model outputs for **M2** (code + original/prefix comment). |
| `llm_analyzed_prefix_method_postfix_comments_deepseek.json` | Model outputs for **M3** (code + updated/postfix comment). |
| `llm_analyzed_prefix_method_strengthened_comment_deepseek.json` | Model outputs for **M4** (code + strengthened comment). |
| `llm_bug_analysis_final_step2_3_pre_postfix_strengthener_comments.py` | Script that builds prompts, calls the LLM API, and writes analysis JSON (same schema as the `llm_analyzed_*.json` files). |

All five JSON files contain **223 aligned entries** (same ordering and `(bug_id, method_name)` keys).

---

## Task: bug detection

For each dataset entry, the model is asked whether the **prefix (pre-fix) focal method** is buggy. The prefix version is taken from the parent commit of a real bug-fix; by construction it is treated as **ground-truth buggy** for this study.

The model must answer in structured JSON with:

- `method_is_buggy`: `"Yes"` or `"No"` (primary prediction),
- `buggy_code_lines`: line references in the numbered code block (model-reported localization),
- `rationale`: short technical explanation (without line numbers in the rationale, per prompt design).

Prompt construction (code sanitization, line numbering, and mode-specific documentation) is implemented in `llm_bug_analysis_final_step2_3_pre_postfix_strengthener_comments.py`.

---

## Documentation conditions (four modes)

| Mode | Name | What the model sees |
|------|------|------------------------|
| **M1** | Code only | Prefix method body with **inline comments stripped**; no Javadoc. |
| **M2** | Original comment | Same code + **Javadoc from the buggy (prefix) revision**. |
| **M3** | Updated comment | Same buggy code + **Javadoc from the fixed (postfix) revision** (semantic alignment stress test). |
| **M4** | Strengthened comment | Same buggy code + **strengthened documentation** produced by the Comment Strengthener pipeline. |

Across modes, the **task instruction and JSON output format are fixed**; only the documentation block (and a one-line “analysis mode” descriptor in the prompt) changes.

---

## How bug detection is evaluated

### Ground truth

- **Positive class (buggy):** every prefix focal method in the dataset is labeled **buggy** by definition (derived from bug-fixing commits).
- There is **no** separate set of non-buggy methods in this benchmark; the study measures how often the model agrees with this single-class ground truth.

### Mapping model output to a binary label

From each result entry, read `focal_methods[0].method_is_buggy` and normalize:

- `"Yes"` (case-insensitive) → **predicted buggy** (positive prediction),
- `"No"` → **predicted not buggy** (negative prediction),
- anything else (empty, parse errors, `"Error"`) → treat as **invalid / other** if you report breakdowns.

### Primary aggregate metric

Because ground truth is **all-buggy**, the natural headline metric is **detection rate** (also called **recall of the “buggy” class** in a one-class setting):

\[
\text{Detection rate} = \frac{\#\text{ entries where prediction is “Yes”}}{\#\text{ entries}}
\]

Equivalently, this is the fraction of true positives among all positives when every instance is truly buggy. In this setup it coincides numerically with **accuracy** if you label every entry as buggy in ground truth and score “Yes” as correct.

**Caveat:** Reporting **precision** or **F1** in the usual binary sense is **not meaningful** without genuine non-buggy instances in the ground truth; reviewers should see **detection rate** (or recall of “buggy”) as the primary number.

### Optional analyses

- **Per-mode confusion-style view:** count `Yes` vs `No` (and `other`) per mode; compare modes on the same 223 entries.
- **Cross-mode patterns:** e.g. entries where M2 predicts buggy but M3 does not (documentation-induced flips), implemented by aligning files by index and comparing labels (see `analyze_four_modes_results.py` in the parent project if you ship it).

### What is *not* evaluated automatically

- **Line-level localization** (`buggy_code_lines`) is **not** compared to a developer-provided oracle in this release; it is auxiliary qualitative output from the model.
- **Rationale quality** is for human inspection unless you add a separate annotation protocol.

---

## Repositories in the dataset

Entries come from multiple open-source Java projects (e.g., Guava, Hadoop, Jenkins, SonarQube-related modules); see each entry’s `bug_report.project_name` and `bug_report.file_path`.

---

## Reproducing or extending experiments

1. Place your API credentials as expected by `llm_bug_analysis_final_step2_3_pre_postfix_strengthener_comments.py` (the script uses HTTP requests; configure keys and endpoints in the script or environment as in your lab setup).
2. Run the script with `strengthened_comments_full.json` as input and choose one analysis mode per run; set the output path to a new JSON file to avoid overwriting published results.
3. Keep **decoding parameters** (temperature, top_p, max tokens) fixed across modes when comparing documentation conditions.

The JSON files in this folder are **frozen results** for the paper; treat them as the canonical outputs unless you intentionally regenerate.

---

## Citation

If you use this material, cite your paper and (if applicable) the original project repositories and bug-fix commits referenced by `bug_report.bug_id`.

---

## License

Add a `LICENSE` file at the repository root that matches your institution’s policy. Dataset snippets are derived from open-source projects under their respective licenses; redistribution of full source may require compliance with those licenses.
