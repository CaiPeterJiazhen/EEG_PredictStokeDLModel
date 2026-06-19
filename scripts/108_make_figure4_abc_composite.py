from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_FIG_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片")
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"

ROC_PATH = DESKTOP_FIG_DIR / "figure4a_final_model_roc.png"
CONF_PATH = DESKTOP_FIG_DIR / "figure4b_final_model_confusion_matrix_traditional_style.png"
LOSS_PATH = DESKTOP_FIG_DIR / "figure4d_final_model_loss_curves.png"

OUT_STEM = OUT_DIR / "figure4_abc_final_model_performance_loss_composite"
DESKTOP_OUT_STEM = DESKTOP_FIG_DIR / "figure4_abc_final_model_performance_loss_composite"


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
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


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
    roc = trim_white_border(Image.open(ROC_PATH), padding=20)
    conf = trim_white_border(Image.open(CONF_PATH), padding=20)
    loss = trim_white_border(Image.open(LOSS_PATH), padding=20)

    canvas_w = 5600
    margin_x = 180
    margin_top = 140
    margin_bottom = 130
    gutter_x = 120
    gutter_y = 145
    top_h = 2120
    bottom_h = 1780
    left_w = 2360
    right_w = canvas_w - 2 * margin_x - gutter_x - left_w
    canvas_h = margin_top + top_h + gutter_y + bottom_h + margin_bottom

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(98)

    roc_fit = fit_image(roc, left_w - 140, top_h - 120)
    conf_fit = fit_image(conf, right_w - 100, top_h - 100)
    loss_fit = fit_image(loss, canvas_w - 2 * margin_x - 220, bottom_h - 80)

    left_x = margin_x
    right_x = margin_x + left_w + gutter_x
    top_y = margin_top
    bottom_y = margin_top + top_h + gutter_y

    draw.text((margin_x - 110, margin_top - 60), "A", fill="black", font=label_font)
    paste_center(canvas, roc_fit, left_x + 70, top_y, left_w - 70, top_h)

    draw.text((right_x - 85, margin_top - 60), "B", fill="black", font=label_font)
    paste_center(canvas, conf_fit, right_x + 35, top_y, right_w - 35, top_h)

    draw.text((margin_x - 110, bottom_y - 30), "C", fill="black", font=label_font)
    paste_center(canvas, loss_fit, margin_x + 70, bottom_y, canvas_w - 2 * margin_x - 70, bottom_h)

    save_all(canvas, OUT_STEM)
    save_all(canvas, DESKTOP_OUT_STEM)
    print(f"saved={OUT_STEM.with_suffix('.png')}")
    print(f"desktop={DESKTOP_OUT_STEM.with_suffix('.png')}")
    print(f"canvas={canvas.size}")


if __name__ == "__main__":
    main()
