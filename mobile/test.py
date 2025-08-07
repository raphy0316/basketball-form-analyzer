from PIL import Image, ImageDraw, ImageFont

# Text to display
text = """The target should always be at the center of the screen,
meaning it should always not exceed the maximum width of the screen,
as the current resize plugin doesn't support full screen frame resizing."""

# Create an image
image_width = 800
image_height = 400
background_color = (30, 30, 30)  # Dark background
text_color = (255, 255, 255)  # White text

# Create a blank image
image = Image.new("RGB", (image_width, image_height), background_color)
draw = ImageDraw.Draw(image)

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
except IOError:
    font = ImageFont.load_default()  # Fallback to default font

# Calculate text position using textbbox
text_bbox = draw.textbbox((0, 0), text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]
text_x = (image_width - text_width) // 2
text_y = (image_height - text_height) // 2

# Add text to the image
draw.multiline_text((text_x, text_y), text, fill=text_color, font=font, align="center")

# Save the image
image.save("notes_image.png")

# Show the image (optional)
image.show()