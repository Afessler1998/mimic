# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import numpy as np
from PIL import Image

SQUARE_SIZE_MM = 25
CHESS_ROWS = 7
CHESS_COLS = 10

DPI = 300
MM_PER_INCH = 25.4
PT_PER_INCH = 72

# US Letter, landscape, since the pattern is wider than it is tall
PAGE_WIDTH_MM = 279.4
PAGE_HEIGHT_MM = 215.9

def pattern_image():
  SQUARE_SIZE_PX = int(SQUARE_SIZE_MM * DPI / MM_PER_INCH)

  pattern_width = CHESS_COLS * SQUARE_SIZE_PX
  pattern_height = CHESS_ROWS * SQUARE_SIZE_PX

  pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)

  for i in range(CHESS_ROWS):
    for j in range(CHESS_COLS):
      if (i + j) % 2 != 0: continue
      y1, y2 = i * SQUARE_SIZE_PX, (i + 1) * SQUARE_SIZE_PX
      x1, x2 = j * SQUARE_SIZE_PX, (j + 1) * SQUARE_SIZE_PX
      pattern[y1:y2, x1:x2] = 255

  border_size = int(SQUARE_SIZE_PX / 2)
  bordered = np.full(
    (pattern_height + 2 * border_size, pattern_width + 2 * border_size),
    255,
    dtype=np.uint8
  )
  bordered[
    border_size:border_size + pattern_height,
    border_size:border_size + pattern_width
  ] = pattern

  return Image.fromarray(bordered)

def mm_to_pt(mm):
  return mm * PT_PER_INCH / MM_PER_INCH

# The square size is what gives triangulation its scale, so the pattern has to
# come off the printer at exactly SQUARE_SIZE_MM. A PDF laid out in points is
# printable at true size, where a PNG is at the mercy of whatever the print
# dialog decides to scale it by.
def write_pdf(path):
  square = mm_to_pt(SQUARE_SIZE_MM)
  page_width = mm_to_pt(PAGE_WIDTH_MM)
  page_height = mm_to_pt(PAGE_HEIGHT_MM)

  origin_x = (page_width - CHESS_COLS * square) / 2
  origin_y = (page_height - CHESS_ROWS * square) / 2

  # PDF puts the origin bottom left, so rows are drawn from the bottom up. The
  # board is symmetric under that flip, but the parity stays explicit anyway.
  squares = []
  for i in range(CHESS_ROWS):
    for j in range(CHESS_COLS):
      if (i + j) % 2 == 0: continue
      x = origin_x + j * square
      y = origin_y + (CHESS_ROWS - 1 - i) * square
      squares.append(f"{x:.4f} {y:.4f} {square:.4f} {square:.4f} re")

  content = ("0 g\n" + "\n".join(squares) + "\nf\n").encode("ascii")

  objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    (
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
      f"{page_width:.4f} {page_height:.4f}] /Contents 4 0 R >>"
    ).encode("ascii"),
    b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
      + content + b"endstream",
  ]

  out = bytearray(b"%PDF-1.4\n")
  offsets = []
  for number, body in enumerate(objects, start=1):
    offsets.append(len(out))
    out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

  xref_offset = len(out)
  out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
  out += b"0000000000 65535 f \n"
  for offset in offsets:
    out += f"{offset:010d} 00000 n \n".encode("ascii")

  out += (
    f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    f"startxref\n{xref_offset}\n%%EOF\n"
  ).encode("ascii")

  with open(path, "wb") as f:
    f.write(out)

def main():
  pattern_image().save('chessboard_pattern.png', dpi=(DPI, DPI))
  write_pdf('chessboard_pattern.pdf')

  print(f"{CHESS_COLS}x{CHESS_ROWS} squares, "
        f"{CHESS_COLS - 1}x{CHESS_ROWS - 1} inner corners, "
        f"{SQUARE_SIZE_MM}mm each")
  print("print the pdf at 100 percent scale, no fit to page")

if __name__ == "__main__":
  main()
