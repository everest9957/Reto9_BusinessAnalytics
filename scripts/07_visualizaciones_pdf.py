# ============================================================
# RETO 9 — Combinar todas las visualizaciones en un PDF
# ============================================================

from PIL import Image
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUTA_FIGURAS = BASE / "salidas" / "figuras"
RUTA_DOCS = BASE / "docs"
RUTA_DOCS.mkdir(parents=True, exist_ok=True)

RUTA_PDF = RUTA_DOCS / "Visualizaciones_Dashboard_BA_Reto9_JuditGiravent.pdf"

print("=" * 60)
print("COMBINANDO VISUALIZACIONES EN PDF")
print("=" * 60)

# Buscar todas las figuras PNG
figuras = sorted(RUTA_FIGURAS.glob("*.png"))
print(f"\nFiguras encontradas: {len(figuras)}")

if not figuras:
    print("❌ No hay figuras para combinar")
    exit(1)

# Abrir todas como imágenes RGB
imagenes = []
for f in figuras:
    img = Image.open(f).convert("RGB")
    imagenes.append(img)
    print(f"  ✓ {f.name} ({img.size[0]}x{img.size[1]})")

# Guardar todas en un PDF
imagenes[0].save(
    RUTA_PDF,
    save_all=True,
    append_images=imagenes[1:],
    resolution=100.0
)

tamanio = RUTA_PDF.stat().st_size / 1024 / 1024
print(f"\n✅ PDF creado: {RUTA_PDF}")
print(f"   Tamaño: {tamanio:.2f} MB")
print(f"   Páginas: {len(imagenes)}")

print("\n" + "=" * 60)
print("✅ VISUALIZACIONES COMBINADAS")
print("=" * 60)