"""Generate the app logo: a Jabulani-style football.

The 2010 World Cup ball's signature is four curved panels sweeping out from
the centre in a pinwheel. That silhouette survives being shrunk to a 48px
home-screen icon, which a photo-realistic ball would not, so it is drawn as
flat curved strokes over a shaded sphere.

Both outputs come from the same geometry so the crisp in-page SVG and the
rasterised launcher icons cannot drift apart:

    bettingedge/web/logo.svg        vector, used for the header mark and favicon
    bettingedge/web/icon-192.png    home-screen launcher
    bettingedge/web/icon-512.png    splash / high density

Deliberately generic: the panel shape and colours evoke the ball, with no
manufacturer name, logo or wordmark anywhere.

    python tools/make_logo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WEB = Path(__file__).resolve().parent.parent / "bettingedge" / "web"

# Panel colours. One is the app's own accent so the icon ties to the interface;
# the rest echo the original ball's red, amber and blue flashes.
PANELS = ["#e5453d", "#f5b544", "#35d0a5", "#4a86e8"]
SEAM = "#131c26"
BALL_LIGHT = (255, 255, 255)
BALL_SHADE = (206, 216, 226)
BACKDROP = (11, 15, 20)

SUPERSAMPLE = 6          # draw large, shrink down: cheap anti-aliasing


def _blade_spine(cx: float, cy: float, r: float, turn: int,
                 steps: int = 56) -> list[tuple[float, float]]:
    """The spine of one panel: a spiral arm curling out from the centre.

    The ball's signature is four blades sweeping outward and around, not
    stripes across the face. A spiral gives that directly — radius grows while
    the angle advances — where a shallow arc just reads as a band.
    """
    base = math.radians(90 * turn)
    span = math.radians(88)
    points = []
    for index in range(steps + 1):
        t = index / steps
        angle = base + span * t
        radius = r * (0.07 + 0.86 * t)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _band(spine: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """A filled band around a curve: narrow at the hub, broad toward the rim.

    Thick lines are drawn as polygons rather than with a wide stroke, because
    PIL's thick-line joints throw off visible spurs when the segments are short
    relative to the width — which they are on a tight curve.
    """
    left, right = [], []
    for index, (x, y) in enumerate(spine):
        ahead = spine[min(index + 1, len(spine) - 1)]
        behind = spine[max(index - 1, 0)]
        dx, dy = ahead[0] - behind[0], ahead[1] - behind[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        t = index / (len(spine) - 1)
        # Grows from a point at the hub, fattest around three-quarters out,
        # then rounds off at the tip.
        taper = (t ** 0.5) * (1.0 - t ** 5) ** 0.35
        half = width * 0.5 * taper
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    return left + right[::-1]


def _sphere(size: int) -> Image.Image:
    """A white ball with off-centre shading, so it reads as round not flat."""
    grid = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(grid, grid)
    # Light from the upper left.
    lit = np.clip(1.0 - (((x + 0.42) ** 2 + (y + 0.42) ** 2) / 3.1), 0.0, 1.0) ** 0.7
    shade = np.stack([
        BALL_SHADE[channel] + (BALL_LIGHT[channel] - BALL_SHADE[channel]) * lit
        for channel in range(3)
    ], axis=-1)
    return Image.fromarray(shade.astype("uint8"), mode="RGB")


def render_png(size: int, backdrop: bool = True) -> Image.Image:
    big = size * SUPERSAMPLE
    canvas = Image.new("RGBA", (big, big), (*BACKDROP, 255) if backdrop else (0, 0, 0, 0))

    centre = big / 2
    radius = big * 0.415

    # Ball body: shaded sphere clipped to a circle.
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius], fill=255)
    canvas.paste(_sphere(big), (0, 0), mask)

    ball = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ball)

    spines = [_blade_spine(centre, centre, radius, turn) for turn in range(4)]
    # Dark groove first, then the colour inset within it, so every panel has a
    # defined edge at any size.
    for spine in spines:
        draw.polygon(_band(spine, big * 0.178), fill=SEAM)
    for spine, colour in zip(spines, PANELS):
        draw.polygon(_band(spine, big * 0.148), fill=colour)

    # Clip the panels to the ball, so nothing bleeds past the edge.
    ball.putalpha(Image.composite(ball.getchannel("A"),
                                  Image.new("L", (big, big), 0), mask))
    canvas.alpha_composite(ball)

    # Outline last, on top of everything.
    ImageDraw.Draw(canvas).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        outline=SEAM, width=int(big * 0.022))

    return canvas.resize((size, size), Image.LANCZOS).filter(
        ImageFilter.UnsharpMask(radius=1, percent=40, threshold=3))


def _points(polygon: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in polygon)


def render_svg(size: int = 64) -> str:
    """Same geometry as vector, for the header mark and the favicon."""
    centre = size / 2
    radius = size * 0.44
    seam_width = size * 0.178
    panel_width = size * 0.148

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size:g} {size:g}" '
        f'role="img" aria-label="football">',
        '<defs>',
        f'<radialGradient id="ball" cx="32%" cy="30%" r="78%">',
        f'<stop offset="0" stop-color="rgb{BALL_LIGHT}"/>',
        f'<stop offset="1" stop-color="rgb{BALL_SHADE}"/>',
        '</radialGradient>',
        f'<clipPath id="edge"><circle cx="{centre:g}" cy="{centre:g}" r="{radius:g}"/></clipPath>',
        '</defs>',
        f'<circle cx="{centre:g}" cy="{centre:g}" r="{radius:g}" fill="url(#ball)"/>',
        '<g clip-path="url(#edge)">',
    ]
    # Same tapered polygons as the raster path, so the two cannot drift apart.
    spines = [_blade_spine(centre, centre, radius, turn, steps=40) for turn in range(4)]
    for spine in spines:
        parts.append(f'<polygon points="{_points(_band(spine, seam_width))}" '
                     f'fill="{SEAM}" stroke="none"/>')
    for spine, colour in zip(spines, PANELS):
        parts.append(f'<polygon points="{_points(_band(spine, panel_width))}" '
                     f'fill="{colour}" stroke="none"/>')
    parts += [
        '</g>',
        f'<circle cx="{centre:g}" cy="{centre:g}" r="{radius:g}" fill="none" '
        f'stroke="{SEAM}" stroke-width="{size * 0.023:.2f}"/>',
        '</svg>',
    ]
    return "".join(parts)


def main() -> None:
    WEB.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        render_png(size).convert("RGB").save(WEB / f"icon-{size}.png", optimize=True)
    # Transparent version for anywhere the dark tile is not wanted.
    render_png(512, backdrop=False).save(WEB / "logo-512.png", optimize=True)
    (WEB / "logo.svg").write_text(render_svg(), encoding="utf-8")
    print(f"wrote icon-192.png, icon-512.png, logo-512.png and logo.svg into {WEB}")


if __name__ == "__main__":
    main()
