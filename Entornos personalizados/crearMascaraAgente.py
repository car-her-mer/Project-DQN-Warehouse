from PIL import Image
import numpy as np

# Cargar la imagen del agente
path_agente = "IA\\Project_RL-Warehouse\\Assets\\agente.png"
agente_img = Image.open(path_agente).convert("RGBA")

# Convertir la imagen del agente a una máscara binaria
# Donde las áreas no transparentes (alfa > 0) son la parte activa del agente.
agente_data = np.array(agente_img)

# Crear una máscara binaria: 1 para la parte activa del agente, 0 para la parte transparente
mask_agente = agente_data[:, :, 3] > 0  # Solo la capa alfa (transparencia)

# Guardar la máscara de agente para verificar visualmente
mask_agente_img = Image.fromarray((mask_agente * 255).astype(np.uint8))
mask_agente_img.show()
