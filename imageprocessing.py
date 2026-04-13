# =============================================================================
#  image_processing.py
#  AI-Based Wardrobe Recommendation System
#  Module 1: Image Processing — Loading, Normalization & Basic Handling
# =============================================================================
#
#  WHY NORMALIZATION?
#  ------------------
#  Raw images store pixel values as integers from 0 to 255.
#  ML models train faster and more stably when inputs are small, consistent
#  numbers. Dividing every pixel by 255.0 squeezes values into [0.0, 1.0].
#
#  HOW IT HELPS ML MODELS:
#  • Prevents any single channel (R/G/B) from dominating others
#  • Speeds up gradient descent (optimizer finds the minimum faster)
#  • Matches the expected input range of most pre-trained models
#    (e.g. MobileNet, ResNet expect values in [0,1] or [-1,1])
#  • Reduces risk of numerical overflow / vanishing gradients
#
# =============================================================================

import cv2          # OpenCV – image reading, resizing, saving
import numpy as np  # NumPy  – array math


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the image from disk
# ─────────────────────────────────────────────────────────────────────────────
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image file and return it as a NumPy array (BGR format).

    Parameters
    ----------
    image_path : str
        Path to the image file (JPG / PNG / WEBP).

    Returns
    -------
    np.ndarray
        Image array with shape (height, width, 3) and dtype uint8.
        Raises FileNotFoundError if the path is invalid.
    """
    image = cv2.imread(image_path)          # Read image (OpenCV loads as BGR)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image from: '{image_path}'\n"
            "Check that the file exists and the path is correct."
        )

    print(f"[✔] Image loaded successfully from '{image_path}'")
    print(f"    Shape  : {image.shape}   (height × width × channels)")
    print(f"    Dtype  : {image.dtype}")
    print(f"    Min px : {image.min()}   Max px : {image.max()}")
    return image


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Resize to the standard model input size (224×224)
# ─────────────────────────────────────────────────────────────────────────────
def resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize the image to target_size using INTER_AREA interpolation
    (best quality when shrinking).

    Parameters
    ----------
    image       : input image array
    target_size : (width, height) tuple — default 224×224 for most CNN models

    Returns
    -------
    np.ndarray — resized image
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    print(f"\n[✔] Image resized to {target_size[1]}×{target_size[0]} pixels")
    return resized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Normalize pixel values from [0, 255] → [0.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert pixel values from uint8 [0, 255] to float32 [0.0, 1.0].

    Division by 255.0 is the simplest and most common normalization.
    More advanced approaches (z-score / ImageNet mean-std subtraction)
    can be applied on top of this for fine-tuned models.

    Parameters
    ----------
    image : uint8 image array

    Returns
    -------
    np.ndarray with dtype float32 and values in [0.0, 1.0]
    """
    # ── BEFORE normalization ──
    print("\n── Before Normalization ──────────────────────────────")
    print(f"   dtype        : {image.dtype}")
    print(f"   pixel range  : [{image.min()}, {image.max()}]")
    print(f"   sample pixel (top-left corner): {image[0, 0]}")

    # ── Normalize ──
    normalized = image.astype(np.float32) / 255.0   # Scale to [0, 1]

    # ── AFTER normalization ──
    print("\n── After Normalization ───────────────────────────────")
    print(f"   dtype        : {normalized.dtype}")
    print(f"   pixel range  : [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"   sample pixel (top-left corner): {normalized[0, 0]}")

    print("\n[✔] Normalization complete — pixels now in [0.0, 1.0]")
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Convert BGR → RGB (most ML frameworks expect RGB)
# ─────────────────────────────────────────────────────────────────────────────
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    OpenCV loads images in BGR channel order.
    TensorFlow / PyTorch / PIL all expect RGB.
    This function flips the channel order.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("[✔] Converted BGR → RGB channel order")
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Expand dimensions for model input (add batch axis)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_for_model(image: np.ndarray) -> np.ndarray:
    """
    ML models expect a batch of images: shape (batch, height, width, channels).
    For a single image, the batch size = 1.

    np.expand_dims adds that extra dimension:
        (224, 224, 3)  →  (1, 224, 224, 3)
    """
    batch = np.expand_dims(image, axis=0)   # Insert batch dimension at index 0
    print(f"[✔] Batch shape ready for model input: {batch.shape}")
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Optionally save the normalized image back to disk
# ─────────────────────────────────────────────────────────────────────────────
def save_normalized_image(normalized: np.ndarray, output_path: str) -> None:
    """
    Convert float32 image back to uint8 (multiply by 255) and save.
    This is only for visualization/debugging — ML models use the float array.
    """
    # Scale back to [0, 255] for saving
    save_img = (normalized * 255).astype(np.uint8)
    cv2.imwrite(output_path, save_img)
    print(f"[✔] Normalized image saved to '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_wardrobe_image(image_path: str, save_output: bool = False) -> np.ndarray:
    """
    Full image processing pipeline:
      Load → Resize → BGR→RGB → Normalize → Prepare for model

    Returns the normalized float32 array ready to pass into a model.
    """
    print("=" * 55)
    print("  WARDROBE IMAGE PROCESSING PIPELINE")
    print("=" * 55)

    # 1. Load
    image = load_image(image_path)

    # 2. Resize to 224×224 (standard CNN input size)
    image = resize_image(image, target_size=(224, 224))

    # 3. Convert BGR → RGB
    image = bgr_to_rgb(image)

    # 4. Normalize
    normalized = normalize_image(image)

    # 5. Optionally save for inspection
    if save_output:
        save_normalized_image(normalized, "normalized_output.jpg")

    # 6. Expand dims for model
    model_input = prepare_for_model(normalized)

    print("\n[✔] Pipeline complete. Returning model-ready array.")
    print("=" * 55)

    # Return the normalized array (without batch dim) for downstream use
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── DEMO WITH A SYNTHETIC IMAGE ──────────────────────────────────────────
    # (Replace "sample_outfit.jpg" with any real image path on your system)
    # We create a synthetic 300×400 image for demonstration purposes.

    print("\n[INFO] No real image provided — generating a synthetic demo image.\n")

    # Create a random RGB image (simulating a clothing photo)
    synthetic_image = np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8)
    demo_path = "demo_outfit.jpg"
    cv2.imwrite(demo_path, synthetic_image)
    print(f"[INFO] Synthetic demo image saved as '{demo_path}'")

    # Run the full pipeline
    result = process_wardrobe_image(demo_path, save_output=True)

    print(f"\nFinal array shape : {result.shape}")
    print(f"Final dtype        : {result.dtype}")
    print(f"Value range        : [{result.min():.4f}, {result.max():.4f}]")

    # ── HOW TO USE WITH A REAL IMAGE ─────────────────────────────────────────
    # result = process_wardrobe_image("path/to/your/outfit.jpg")
    # model.predict(np.expand_dims(result, axis=0))
