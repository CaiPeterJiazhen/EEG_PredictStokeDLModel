from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_FIG_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片")
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"

SCORECARD_PATH = DESKTOP_FIG_DIR / "patient_level_model_scorecard_with_barlow.png"
SEED_PATH = DESKTOP_FIG_DIR / "figure5a_seed_stability.png"
BRIER_PATH = DESKTOP_FIG_DIR / "figure4c_b_brier_calibration.png"

OUT_STEM = OUT_DIR / "figure5_abc_model_scorecard_seed_brier_composite"
DESKTOP_OUT_STEM = DESKTOP_FIG_DIR / "figure5_abc_model_scorecard_seed_brier_composite"


def trim_white_border(image: Image.Image, *, padding: int = 18, threshold: int = 248) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, background).convert("L")
    mask = diff.point(lambda p: 255 if p < 255 - threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb.width, right + padding)
    bottom = min(rgb.height, bottom + padding)
    return rgb.crop((left, top, right, bottom))


def fit_image(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / image.width, box_h / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def remove_existing_panel_label(image: Image.Image, kind: str) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    if kind in {"scorecard", "seed"}:
        draw.rectangle((0, 0, 135, 150), fill="white")
    elif kind == "brier":
        draw.rectangle((0, 0, 500, 205), fill="white")
    return image


def paste_center(canvas: Image.Image, image: Image.Image, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    px = x + (w - image.width) // 2
    py = y + (h - image.height) // 2
    canvas.paste(image, (px, py))
    return px, py, image.width, image.height


def save_all(image: Image.Image, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    image.save(stem.with_suffix(".png"), dpi=(600, 600))
    image.save(stem.with_suffix(".pdf"), resolution=600.0)


def main() -> None:
    scorecard = trim_white_border(remove_existing_panel_label(Image.open(SCORECARD_PATH), "scorecard"), padding=20)
    seed = trim_white_border(remove_existing_panel_label(Image.open(SEED_PATH), "seed"), padding=20)
    brier = trim_white_border(remove_existing_panel_label(Image.open(BRIER_PATH), "brier"), padding=20)

    canvas_w = 6000
    margin_x = 170
    margin_top = 135
    margin_bottom = 130
    gutter_y = 130
    gutter_x = 170
    top_h = 2250
    bottom_h = 2200
    bottom_w = (canvas_w - 2 * margin_x - gutter_x) // 2
    canvas_h = margin_top + top_h + gutter_y + bottom_h + margin_bottom

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(92)

    score_fit = fit_image(scorecard, canvas_w - 2 * margin_x - 170, top_h - 90)
    seed_fit = fit_image(seed, bottom_w - 80, bottom_h - 90)
    brier_fit = fit_image(brier, bottom_w - 80, bottom_h - 90)

    top_x = margin_x
    top_y = margin_top
    bottom_y = margin_top + top_h + gutter_y
    left_x = margin_x
    right_x = margin_x + bottom_w + gutter_x

    draw.text((margin_x - 115, margin_top - 55), "A", fill="black", font=label_font)
    paste_center(canvas, score_fit, top_x + 60, top_y, canvas_w - 2 * margin_x - 60, top_h)

    draw.text((margin_x - 115, bottom_y - 45), "B", fill="black", font=label_font)
    paste_center(canvas, seed_fit, left_x + 50, bottom_y, bottom_w - 50, bottom_h)

    draw.text((right_x - 95, bottom_y - 45), "C", fill="black", font=label_font)
    paste_center(canvas, brier_fit, right_x + 70, bottom_y, bottom_w - 70, bottom_h)

    save_all(canvas, OUT_STEM)
    save_all(canvas, DESKTOP_OUT_STEM)
    print(f"saved={OUT_STEM.with_suffix('.png')}")
    print(f"desktop={DESKTOP_OUT_STEM.with_suffix('.png')}")
    print(f"canvas={canvas.size}")


if __name__ == "__main__":
    main()
