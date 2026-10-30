# %% Local Grad-CAM previews of the test subset (CPU, no TensorFlow, no model, no inference): drawn from the stored maps, the target images, the target expert masks and the stored pixel predictions
#
# Reads reports/tables/gradcam_test_*.csv and data/processed/gradcam_test_maps.npz (written by analysis/10_interpretation.py), the TARGET images, the expert masks of the TARGET images (reference contour
# only) and the stored pixel predictions of the automatic segmentation (reference only; nothing is recomputed). It writes the preview figures to data/processed/. The previews contain test images and stay local.
#
# Per figure at most two rows: the original with labelled reference contours and boxes, then the Grad-CAM of the full-image, expert-ROI and automatic-ROI condition on the same original image. Every map is
# normalised by its OWN positive maximum (colours are no absolute comparison); outside the ROI the map is NOT COMPUTED (washed out, not zero attention); an empty map is marked and not coloured. The figures
# describe a few rule-selected examples: no frequency, no ranking, no localisation metric, no cause, no clinical statement. The visual inspection is done by the project owner.
#
# One renderer, two layouts: "standard" and "readable" (more room for legend and footnote, written to a separate file *_readable.png for one group).
# Importing this module draws nothing.
# Usage (from the project root BUSBRA): .venv/bin/python plots/plot_gradcam_test_interpretation.py [--readable [group name, default core_FN]]
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from busbra import auto_roi as ar  # noqa: E402
from busbra import data, gradcam_test as gt, image_processing as ip, results, roi  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MAPS_PATH, PIXELS = PROCESSED / "gradcam_test_maps.npz", PROCESSED / "test_unet_predicted_masks.npz"
CONDITIONS = ("full", "expert", "auto")
CONDITION_LABEL = {"full": "Full image (4B)", "expert": "Expert-mask ROI (4E)", "auto": "Automatic ROI (4F-B boxes, 4E classifier)"}
SPECIAL_LABEL = {"fallback_full_image": "empty segmentation: fallback = whole image", "crop_without_expert_pixel": "automatic crop contains no expert-mask pixel",
                 "chosen_component_misses_but_other_overlaps": "chosen component misses the lesion, another component overlaps it", "crop_covers_below_90_percent": "automatic crop covers 0 < x < 90 % of the expert-mask pixels"}
# the two layouts differ only in the room for legend and footnote: figure height, legend anchor, footnote wrap/size/position, subplot margins
LAYOUTS = {"standard": {"extra_height": 1.9, "legend_y": 0.045, "note_wrap": 205, "note_size": 7.2, "note_y": 0.006, "top": (0.85, 0.89), "bottom": (0.17, 0.14)},
           "readable": {"extra_height": 3.4, "legend_y": 0.125, "note_wrap": 170, "note_size": 7.6, "note_y": 0.012, "top": (0.86, 0.90), "bottom": (0.30, 0.24)}}
EXPERT_BOX_COLOR, AUTO_BOX_COLOR, CONTOUR_COLOR, PRED_COLOR = "#FFD700", "#00E5FF", "#39FF14", "#FF00FF"


def load_sources() -> dict:
    """The stored selection, geometry and Grad-CAM tables and the stored maps."""
    tables = {n: results.table(f"gradcam_test_{n}", keep_default_na=False) for n in ("selection", "geometry", "cam")}
    with np.load(MAPS_PATH, allow_pickle=False) as archive:
        maps = {k: archive[k] for k in archive.files}
    problems = gt.table_problems(tables["selection"], tables["geometry"], tables["cam"], maps, tables["selection"]["ID"])
    if problems:
        raise ValueError("the stored Grad-CAM tables do not match the renderer: " + "; ".join(problems[:5]))
    return {"tables": tables, "maps": maps}


def preview_groups(sources: dict) -> list:
    """Figure groups (name, ids) from the stored selection: one figure per core category (at most two rows), the special cases in figures of at most two rows."""
    rows = sources["tables"]["selection"].to_dict("records")
    groups = gt.preview_groups([{"ID": r["ID"], "role": r["role"], "category": r["category"]} for r in rows])
    for group in groups:
        group["path"] = PROCESSED / f"gradcam_test_preview_{group['name']}.png"
    return groups


def load_references(targets: list) -> dict:
    """Target images, target expert masks and the stored pixel predictions of the targets (read for display only)."""
    manifest = data.read_manifest()
    by_id = {r["ID"]: r for r in data.subset_frame(manifest, "test").to_dict("records")}
    images, masks = {}, {}
    for identifier in targets:
        row = by_id[identifier]
        images[identifier] = ip.load_grayscale_image(ROOT / row["image_relpath"], (int(row["image_width"]), int(row["image_height"])))
        description, array = roi.read_mask_raw(ROOT / row["mask_relpath"])
        masks[identifier] = roi.decode_mask(description, array, expected_size=images[identifier].size)
    with np.load(PIXELS, allow_pickle=False) as archive:
        predicted = ar.unpack_masks({k: archive[k] for k in ("ids", "shapes", "offsets", "bits")})
    return {"images": images, "masks": masks, "predicted": {i: predicted[i] for i in targets}, "manifest_rows": {i: by_id[i] for i in targets}}


def draw_box(ax, box, color, style, width):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor=color, linestyle=style, linewidth=width))


def draw_group(group: dict, sources: dict, refs: dict, path: Path, layout: str = "standard") -> dict:
    lay = LAYOUTS[layout]
    tables = sources["tables"]
    selection = tables["selection"].set_index("ID", drop=False)
    cam = tables["cam"].set_index(["condition", "ID"], drop=False)
    geometry = tables["geometry"].set_index(["condition", "ID"], drop=False)
    maps = sources["maps"]
    n = len(group["ids"])
    fig, axes = plt.subplots(n, 4, figsize=(16.5, 4.9 * n + lay["extra_height"]), squeeze=False)
    wash = ListedColormap(["white"])
    for r, identifier in enumerate(group["ids"]):
        s = selection.loc[identifier]
        image = np.asarray(refs["images"][identifier], dtype=np.float32)
        height, width = image.shape
        mask, pred = refs["masks"][identifier], refs["predicted"][identifier]
        box_of = {c: tuple(int(geometry.loc[(c, identifier), k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")) for c in CONDITIONS}
        ax = axes[r][0]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.contour(mask.astype(float), levels=[0.5], colors=CONTOUR_COLOR, linewidths=1.3)
        if pred.any():
            ax.contour(pred.astype(float), levels=[0.5], colors=PRED_COLOR, linestyles=":", linewidths=1.3)
        draw_box(ax, box_of["expert"], EXPERT_BOX_COLOR, "--", 1.4)
        draw_box(ax, box_of["auto"], AUTO_BOX_COLOR, "-", 1.4)
        title = gt.case_title_lines(s, SPECIAL_LABEL)
        ax.set_title("\n".join([title[0], textwrap.shorten(title[1], 38), textwrap.fill(title[2], 46)]), fontsize=8, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(-0.5, width - 0.5), ax.set_ylim(height - 0.5, -0.5)
        for c, condition in enumerate(CONDITIONS, start=1):
            ax = axes[r][c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            entry = cam.loc[(condition, identifier)]
            heat = maps[f"image__{condition}__{identifier}"]
            ax.imshow(np.where(np.isnan(heat), 1.0, np.nan), cmap=wash, vmin=0, vmax=1, alpha=0.62)  # outside the ROI: washed out = not computed
            empty = str(entry["empty_after_relu"]) == "True"
            if not empty:
                ax.imshow(np.ma.masked_invalid(heat), cmap="jet", vmin=0, vmax=1, alpha=0.5)
            else:
                ax.text(0.5, 0.5, "EMPTY MAP\n(no positive value after ReLU)", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="black", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"})
            if condition == "expert":
                draw_box(ax, box_of[condition], EXPERT_BOX_COLOR, "--", 1.4)
            elif condition == "auto":
                draw_box(ax, box_of[condition], AUTO_BOX_COLOR, "-", 1.4)
            ax.contour(mask.astype(float), levels=[0.5], colors=CONTOUR_COLOR, linewidths=0.8)
            lines = gt.panel_lines(s, entry, condition, CONDITION_LABEL[condition])
            ax.set_title("\n".join(textwrap.fill(line, 52) for line in lines), fontsize=7.6, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim(-0.5, width - 0.5), ax.set_ylim(height - 0.5, -0.5)
    fig.suptitle(f"Step 5D, test subset, seed 7: Grad-CAM (layer mobilenetv2_backbone/Conv_1, 5 x 5, target = malignant logit) - {group['name'].replace('_', ' ')}", fontsize=10, y=0.995)
    handles = [Line2D([0], [0], color=CONTOUR_COLOR, lw=1.6, label="expert mask contour (reference only)"), Line2D([0], [0], color=EXPERT_BOX_COLOR, lw=1.6, ls="--", label="expert ROI box (4E classifier input)"),
               Line2D([0], [0], color=AUTO_BOX_COLOR, lw=1.6, label="automatic ROI box (frozen 5C; = whole image for a fallback)"), Line2D([0], [0], color=PRED_COLOR, lw=1.6, ls=":", label="all predicted U-Net pixels (frozen 5C, reference only)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=True, bbox_to_anchor=(0.45, lay["legend_y"]))
    note = ("Each map is normalised by its OWN positive maximum: colours are no absolute comparison between panels or models. Washed-out area = outside the ROI: not computed (not 'zero attention'). Decisions use each condition's own frozen "
            "validation threshold (0.95 sensitivity target). The category refers to the automatic branch. Maps show where the gradient-weighted activation is, not why; no cause, no clinical correctness. Local preview with test images.")
    fig.text(0.01, lay["note_y"], textwrap.fill(note, lay["note_wrap"]), fontsize=lay["note_size"], va="bottom")
    cax = fig.add_axes([0.945, 0.30, 0.01, 0.40])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet"), cax=cax).set_label("relative map value (own maximum = 1)", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.93, top=lay["top"][n > 1], bottom=lay["bottom"][n > 1], wspace=0.05, hspace=0.40)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return {"rows": n, "size_bytes": path.stat().st_size}


def main(argv: list[str]) -> int:
    readable = "--readable" in argv
    names = [a for a in argv if not a.startswith("--")]
    sources = load_sources()
    groups = preview_groups(sources)
    if readable:
        group = {g["name"]: g for g in groups}[names[0] if names else "core_FN"]
        target = PROCESSED / f"gradcam_test_preview_{group['name']}_readable.png"
        draw_group(group, sources, load_references(list(group["ids"])), target, "readable")
        print(f"wrote {target.relative_to(ROOT).as_posix()}")
        return 0
    refs = load_references([r["ID"] for r in sources["tables"]["selection"].to_dict("records")])
    for group in groups:
        draw_group(group, sources, refs, group["path"])
    print(f"wrote {len(groups)} previews to data/processed/ (they contain test images and stay local)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if "__file__" in globals() else []))
