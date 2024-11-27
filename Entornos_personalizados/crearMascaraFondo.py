from PIL import Image
import numpy as np

# Cargar las imágenes proporcionadas
path_fondo = "IA\\Project_RL-Warehouse\\Assets\\fondo.png"

# Cargar las imágenes
fondo_img = Image.open(path_fondo).convert("RGBA")

# Convertir el fondo a una máscara binaria
# Todo lo que no sea el área blanca o naranja será obstáculo
fondo_data = np.array(fondo_img)

# Definir los colores de las zonas permitidas (blanco y naranja)
blanco = [255, 255, 255]
naranja_claro = [255, 165, 0]
naranja_oscuro = [255, 140, 0]

# Crear una máscara binaria: 1 para permitido, 0 para obstáculo
mask_permitido = np.all(fondo_data[:, :, :3] == blanco, axis=-1) | \
                 np.all(fondo_data[:, :, :3] == naranja_claro, axis=-1) | \
                 np.all(fondo_data[:, :, :3] == naranja_oscuro, axis=-1)

# Invertir la máscara para obtener obstáculos
mask_obstaculos = ~mask_permitido

# Guardar la máscara de obstáculos para verificar visualmente
mask_obstaculos_img = Image.fromarray((mask_obstaculos * 255).astype(np.uint8))
mask_obstaculos_img.show()