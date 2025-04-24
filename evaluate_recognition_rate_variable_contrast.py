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

def evaluate_recognition_rate(fg_masks, bg_masks, digit, fg_brightness, bg_brightness, noise_variance, trials=100):
    correct_count = 0
    for _ in range(trials):
        # генерация изображения с шумом
        image_path = os.path.join(input_folder, f"{digit}.bmp")
        fg_mask, bg_mask = get_foreground_background(image_path)
        noisy_image = create_image_with_noise(fg_mask, bg_mask, fg_brightness, bg_brightness, noise_variance)

        # распознавание цифры
        input_image = np.array(noisy_image.convert('L'))
        threshold = 127
        input_image_binary = (input_image < threshold).astype(np.uint32)
        recognized_digit = recognize_digit(input_image_binary, fg_masks, bg_masks)

        if recognized_digit == digit:
            correct_count += 1

    return correct_count / trials

if __name__ == "__main__":
    input_folder = 'C:\\Users\\polin\\PycharmProjects\\Raspoznavanie\\numbers'
    fg_masks, bg_masks = create_digit_masks(input_folder)

    # определение диапазона дисперсий и контрастностей
    noise_variances = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    contrasts = [(0, 255), (25, 230), (50, 205), (75, 180), (100, 155), (125, 130)]
    recognition_rates = {contrast: [] for contrast in contrasts}

    for fg_brightness, bg_brightness in contrasts:
        for variance in noise_variances:
            correct_rate = 0
            for digit in range(10):
                rate = evaluate_recognition_rate(fg_masks, bg_masks, digit=digit,
                                                  fg_brightness=fg_brightness,
                                                  bg_brightness=bg_brightness,
                                                  noise_variance=variance,
                                                  trials=100)
                correct_rate += rate
            recognition_rates[(fg_brightness, bg_brightness)].append(correct_rate / 10)

    # построение графиков для каждой контрастности
    plt.figure(figsize=(10, 5))
    for (fg_brightness, bg_brightness), rates in recognition_rates.items():
        plt.plot(noise_variances, rates, marker='o', label=f'Contrast: |{fg_brightness} - {bg_brightness}|')

    plt.title('График зависимости частоты правильного распознавания от дисперсии для различных контрастностей')
    plt.xlabel('Дисперсия шума')
    plt.ylabel('Частота правильного распознавания')
    plt.xscale('log')  # логарифмическая шкала для оси X
    plt.legend()
    plt.grid()
    plt.show()