import numpy as np
import matplotlib.pyplot as plt

def binary_to_checkerboard(binary_image):
    # Compute 2D DFT of the binary image
    dft_binary = np.fft.fft2(binary_image)
    
    # Generate a checkerboard pattern of the same size
    size = binary_image.shape[0]
    checkerboard = generate_checkerboard(size)
    
    # Multiply the magnitude of the binary image's DFT with the checkerboard pattern
    dft_transformed = np.abs(dft_binary) * checkerboard
    
    # Take the inverse 2D DFT to obtain the transformed image
    transformed_image = np.fft.ifft2(dft_transformed).real
    
    return transformed_image

def generate_checkerboard(size):
    # Generate a checkerboard pattern
    checkerboard = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i // 2) % 2 == (j // 2) % 2:
                checkerboard[i, j] = 1
    return checkerboard

# Example: Convert a non-checkerboard binary image to a checkerboard pattern
binary_image = np.array([[0, 1, 1, 0],
                         [0, 1, 0, 1],
                         [1, 0, 1, 0],
                         [0, 1, 0, 1]])

# Convert binary image to checkerboard pattern
transformed_image = binary_to_checkerboard(binary_image)

# Plotting
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Non-Checkerboard Binary Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Checkerboard Pattern')
plt.axis('off')
plt.show()
