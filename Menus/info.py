# Pantalla de información del proyecto.
# Este archivo muestra una imagen con información y un botón para continuar.
# Se usa como pantalla intermedia antes de iniciar el entrenamiento o volver al menú principal.

import pygame
import sys
import os
os.environ['SDL_VIDEO_CENTERED'] = '1'  # ← Esto centra la ventana

def mostrar_submenu(pantalla):
    imagen = pygame.image.load("Assets\\info.png")  
    imagen = pygame.transform.scale(imagen, (1300, 800))

    ancho, alto = imagen.get_size()

    # Crear la ventana del mismo tamaño que la imagen
    pantalla = pygame.display.set_mode((ancho, alto))

    fuente = pygame.font.SysFont("Calibri", 24, bold=True)

    # Botón en esquina inferior izquierda
    boton_ancho = 200
    boton_alto = 50
    boton_x = 20
    boton_y = alto - boton_alto - 20  # Posición vertical del botón (20 píxeles desde abajo)
    boton_rect = pygame.Rect(boton_x, boton_y, boton_ancho, boton_alto)  # Crear el rectángulo del botón

    ejecutando = True  # Controla el bucle principal de la pantalla
    while ejecutando:
        pantalla.fill((30, 30, 30))  # Rellenar el fondo con un color gris oscuro

        # Mostrar imagen de información ocupando toda la ventana
        pantalla.blit(imagen, (0, 0))

        # Dibujar el botón en la esquina inferior izquierda
        pygame.draw.rect(pantalla, (100, 200, 100), boton_rect)  # Botón verde
        texto_boton = fuente.render("Continuar", True, (0, 0, 0))  # Texto del botón en negro

        # Centrar el texto dentro del botón
        texto_boton_rect = texto_boton.get_rect(center=boton_rect.center)
        pantalla.blit(texto_boton, texto_boton_rect)

        pygame.display.flip()  # Actualizar la pantalla para mostrar los cambios

        # Manejar eventos del usuario (teclado, ratón, cerrar ventana)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                ejecutando = False  # Salir si se pulsa ESC
            elif evento.type == pygame.MOUSEBUTTONDOWN:
                if boton_rect.collidepoint(evento.pos):  # Si se hace clic en el botón
                    return "entrenar"  # ← Aquí retorna al menú principal o siguiente acción
