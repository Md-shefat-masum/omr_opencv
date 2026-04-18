import cv2
from matplotlib import pyplot as plt

img = cv2.imread("21.bmp", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Could not read image: 1.jpg")

OUT_W, OUT_H = 930, 620
img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
print(img.shape)
print(f"window size: {OUT_W}x{OUT_H}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dpi = 100
fig, ax = plt.subplots(figsize=(OUT_W / dpi, OUT_H / dpi), dpi=dpi)
ax.imshow(img_rgb)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
