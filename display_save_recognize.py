import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

def get_foreground_background(image_path):
    img = np.array(Image.open(image_path).convert('L'))
    threshold = 127
    fg = (img < threshold).astype(np.uint8)
    bg = (img >= threshold).astype(np.uint8)
    return fg, bg

def create_image(fg_mask, bg_mask, fg_brightness, bg_brightness):
    new_img = fg_brightness * fg_mask + bg_brightness * bg_mask
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return Image.fromarray(new_img)

def create_image_with_noise(fg_mask, bg_mask, fg_brightness, bg_brightness, noise_variance):
    new_img = fg_brightness * fg_mask + bg_brightness * bg_mask
    noise = np.random.normal(0, np.sqrt(noise_variance), new_img.shape)
    noisy_img = new_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def display_images(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Без шума')
    axes[0].axis('off')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('С шумом')
    axes[1].axis('off')
    return fig

def save_images(fig, output_path):
    fig.savefig(output_path, bbox_inches='tight')

def create_digit_masks(input_folder):
    fg_masks = []
    bg_masks = []
    for digit in range(10):
        image_path = os.path.join(input_folder, f"{digit}.bmp")
        try:
            fg, bg = get_foreground_background(image_path)
            fg_masks.append(fg)
            bg_masks.append(bg)
        except Exception as e:
            print(f"Ошибка при обработке цифры {digit}: {e}")
    return fg_masks, bg_masks

def project_image(noisy_image, fg_masks, bg_masks):
    projections = []
    for fg_mask, bg_mask in zip(fg_masks, bg_masks):
        scalar_product_fg_fg = np.sum(fg_mask * fg_mask)
        scalar_product_bg_bg = np.sum(bg_mask * bg_mask)

        scalar_product_fg_f = np.sum(fg_mask * noisy_image)
        scalar_product_bg_f = np.sum(bg_mask * noisy_image)

        c_1 = scalar_product_fg_f / scalar_product_fg_fg if scalar_product_fg_fg != 0 else 0
        c_2 = scalar_product_bg_f / scalar_product_bg_bg if scalar_product_bg_bg != 0 else 0

        projection = c_1 * fg_mask + c_2 * bg_mask
        projections.append(projection)

    return projections

def calculate_squared_norm(projected_image, original_image):
    return np.sum((projected_image - original_image) ** 2)

def recognize_digit(noisy_image, fg_masks, bg_masks):
    projections = project_image(noisy_image, fg_masks, bg_masks)
    min_norm = float('inf')
    recognized_digit = -1

    for i, projection in enumerate(projections):
        norm = calculate_squared_norm(noisy_image, projection)
        if norm < min_norm:
            min_norm = norm
            recognized_digit = i

    return recognized_digit

if __name__ == "__main__":
    input_folder = 'C:\\Users\\polin\\PycharmProjects\\Raspoznavanie\\numbers'
    fg_masks, bg_masks = create_digit_masks(input_folder)

    # генерация случайной цифры
    digit_to_generate = random.randint(0, 9)
    image_path = os.path.join(input_folder, f"{digit_to_generate}.bmp")
    fg_mask, bg_mask = get_foreground_background(image_path)

    img_no_noise = create_image(fg_mask, bg_mask, fg_brightness=0, bg_brightness=255)
    img_with_noise = create_image_with_noise(fg_mask, bg_mask, fg_brightness=0, bg_brightness=255, noise_variance=100000)

    fig = display_images(img_no_noise, img_with_noise)
    save_images(fig, "output_image.png")
    plt.show()

    # распознавание цифры из зашумлённого изображения
    input_image = np.array(img_with_noise.convert('L'))
    threshold = 127
    input_image_binary = (input_image < threshold).astype(np.uint32)

    recognized_digit = recognize_digit(input_image_binary, fg_masks, bg_masks)
    print(f"Распознанная цифра: {recognized_digit}")