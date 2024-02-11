import matplotlib.pyplot as plt
import numpy as np

# Adjusting the figure size and spacing between lines for a more horizontal and flatter appearance
fig, ax = plt.subplots(figsize=(12, 4))

# Hex color codes
blue_color = "#1f77b4"  # Blue
green_color = "#2ca02c"  # Green

# Input range parameters
n_input_lines = 10
x_input = np.linspace(0, 20, 1000)

# Output range parameters
n_output_lines = 3
x_output = np.linspace(20, 30, 500)

# Flattening the squiggly lines by reducing the amplitude
amplitude_reduction = 0.5

# Creating flatter input squiggly lines with specific color
for i in range(n_input_lines):
    y = (
        np.sin(x_input * (0.5 + i / n_input_lines))
        * np.random.uniform(0.3, 0.7)
        * amplitude_reduction
    )
    ax.plot(x_input, y - i * 0.5, color=blue_color)

# Creating flatter output squiggly lines with specific color
for i in range(n_output_lines):
    y = (
        np.sin(x_output * (0.5 + i / n_output_lines))
        * np.random.uniform(0.3, 0.7)
        * amplitude_reduction
    )
    ax.plot(x_output, y - i, color=green_color)

# Adding updated annotations with days and font size 24
ax.text(
    10,
    1,
    "Input Sequence (8 days)",
    horizontalalignment="center",
    fontsize=24,
    color=blue_color,
)
ax.text(
    25,
    1,
    "Output Sequence (4 days)",
    horizontalalignment="center",
    fontsize=24,
    color=green_color,
)

# Adjusting plot aesthetics
ax.axis("off")
plt.tight_layout()

# File paths for saving the images
hex_colors_png_file = "/mnt/data/input_output_sequence_hex_colors.png"
hex_colors_svg_file = "/mnt/data/input_output_sequence_hex_colors.svg"

# Saving as PNG with transparency
fig.savefig(hex_colors_png_file, transparent=True, bbox_inches="tight")

# Saving as SVG (vector image)
fig.savefig(hex_colors_svg_file, format="svg", bbox_inches="tight")
