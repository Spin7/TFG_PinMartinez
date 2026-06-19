import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ["Female", "Male"]
MAX_CROPS = 6

def visualize_result(img, boxes, preds, save_path):

    img_draw = img.copy()
    crops = []

    for box, pred in zip(boxes, preds):
        x1, y1, x2, y2 = map(int, box)

        color = (0,200,0) if pred == 0 else (200,0,0)
        label = CLASS_NAMES[pred]

        # Draw bbox
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label,
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, 1)

        cv2.rectangle(img_draw,
                      (x1, y1-th-5),
                      (x1+tw, y1),
                      color, -1)

        cv2.putText(img_draw,
                    label,
                    (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1)

        # Crop
        crop = img[y1:y2, x1:x2]
        if crop.size != 0:
            crop = cv2.resize(crop, (120,120))
            crops.append((crop, label))

    crops = crops[:MAX_CROPS]

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    cols = max(3, len(crops))
    fig = plt.figure(figsize=(4*cols, 6))

    # Original
    ax1 = plt.subplot(2, cols, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    # Detection
    ax2 = plt.subplot(2, cols, 2)
    ax2.imshow(img_draw_rgb)
    ax2.set_title("Detections")
    ax2.axis("off")

    # Crops
    for i, (crop, label) in enumerate(crops):
        ax = plt.subplot(2, cols, cols + i + 1)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()