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
    boton_y = alto - boton_alto - 20
    boton_rect = pygame.Rect(boton_x, boton_y, boton_ancho, boton_alto)

    ejecutando = True
    while ejecutando:
        pantalla.fill((30, 30, 30))

        # Mostrar imagen
        pantalla.blit(imagen, (0, 0))

        # Mostrar botón
        pygame.draw.rect(pantalla, (100, 200, 100), boton_rect)
        texto_boton = fuente.render("Continuar", True, (0, 0, 0))

        # Centra el texto dentro del botón
        texto_boton_rect = texto_boton.get_rect(center=boton_rect.center)
        pantalla.blit(texto_boton, texto_boton_rect)

        pygame.display.flip()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                ejecutando = False
            elif evento.type == pygame.MOUSEBUTTONDOWN:
                if boton_rect.collidepoint(evento.pos):
                    return "entrenar"  # ← Aquí retorna al menú principal
