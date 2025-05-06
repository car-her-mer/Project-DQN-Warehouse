# Este archivo contiene la lógica para mostrar el menú principal de la aplicación, 
# con botones para ir a otros menús, como el de información del proyecto o el entorno de OpenAI Gym.

# En el menú principal, el usuario puede hacer clic en botones para:
# # Ir al menú de información (carga menu_info.py).
# # Ir al entorno de OpenAI Gym (ejecuta ejecutar_gym.py).

import pygame
import sys

def ajustar_texto(texto, fuente, max_ancho):
    """
    Ajusta el texto para que quepa en varias líneas según el ancho máximo permitido.
    """
    palabras = texto.split(' ')
    lineas = []
    linea_actual = palabras[0]
    
    for palabra in palabras[1:]:
        # Verificar si añadir la palabra sobrepasaría el límite de ancho
        if fuente.size(linea_actual + ' ' + palabra)[0] <= max_ancho:
            linea_actual += ' ' + palabra
        else:
            lineas.append(linea_actual)
            linea_actual = palabra
    
    # Agregar la última línea
    lineas.append(linea_actual)
    
    return lineas

def crear_fuente(pantalla, porcentaje_altura=0.05):
    alto = pantalla.get_height()
    tamaño_fuente = max(16, int(alto * porcentaje_altura))  # tamaño mínimo de fuente
    return pygame.font.SysFont("Calibri", tamaño_fuente)

def mostrar_menu():
    pygame.init()
    pantalla = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Menú Principal")

    fuente = pygame.font.SysFont("Calibri", 48)
    opciones = ["1. Iniciar entrenamiento", "2. Ver información", "3. Salir"]

    seleccion = 0
    ejecutando = True

    while ejecutando:
        fuente = crear_fuente(pantalla)
        pantalla.fill((0, 0, 0))

        for i, opcion in enumerate(opciones):
            color = (255, 255, 255) if i == seleccion else (150, 150, 150)
            texto = fuente.render(opcion, True, color)
            pantalla.blit(texto, (100, 100 + i * 60))

        pygame.display.flip()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_DOWN:
                    seleccion = (seleccion + 1) % len(opciones)
                elif evento.key == pygame.K_UP:
                    seleccion = (seleccion - 1) % len(opciones)
                elif evento.key == pygame.K_RETURN:
                    if seleccion == 0:
                        # ejecutando = False  # Continuar al entorno
                        return "entrenar"
                    elif seleccion == 1:
                        mostrar_informacion(pantalla, fuente)
                    elif seleccion == 2:
                        pygame.quit()
                        sys.exit()

    pygame.quit()


def mostrar_informacion(pantalla, fuente):
    informacion = [
        "(Para hacer scroll, utiliza las flechas del teclado. Para volver dale a ESC.)",
        "",
        "Proyecto Deep Reinforcement Learning - Optimización de rutas.",
        "",
        "1. PROBLEMA.",
        "• El problema que se busca solucionar sería el de optimización de rutas, donde el agente (robot, reparto, uber, etc.) encuentra el camino más eficiente del punto A al punto B.",
        "",
        "• Actualmente se resuelve con tres enfoques tradicionales: Grafos, optimización y heurísticas.",
        "",
        "• OPTIMIZACIÓN (Ejemplo: Dijkstra)",
        "· Concepto: ",
        "La optimización en el contexto de rutas significa encontrar el camino más eficiente entre dos puntos en una red.",
        "",
        "· Ejemplo Conceptual: ",
        "Imagina que el robot tiene un mapa y usa un algoritmo como Dijkstra para saber qué camino tomar.",
        "",
        "· Limitación: ",
        "Este algoritmo supone que el entorno es estático y no tiene capacidad para adaptarse a cambios imprevistos.",
        "",
        "• HEURÍSTICAS (Ejemplo: A*)",
        "· Concepto: ",
        "Una heurística es una regla rápida para ayudar a reducir el tiempo de búsqueda de una solución sin garantizar que sea perfecta.",
        "",
        "· Ejemplo Conceptual: ",
        "El robot usa una heurística, como moverse hacia el objetivo si parece más cerca.",
        "",
        "· Limitación: ",
        "Aunque es más rápido que Dijkstra, A* aún requiere que el entorno sea relativamente estable.",
        "",
        "• GRAFOS",
        "· Concepto: ",
        "Un grafo representa elementos (nodos) y sus conexiones (aristas), y se usa para representar mapas de rutas.",
        "",
        "· Ejemplo Conceptual: ",
        "Imagina el almacén como nodos (estanterías, áreas) conectados por aristas (caminos). Los algoritmos de grafos como Dijkstra buscan el camino más corto.",
        "",
        "· Limitación: ",
        "Los grafos no son dinámicos y necesitan recalcular todo si algo cambia.",
        "",
        "2. IDEA.",
        "• DQN (Deep Q-Learning)",
        "· Concepto General: ",
        "DQN es una técnica de Reinforcement Learning que permite a un agente aprender qué acciones tomar en cada situación para maximizar sus recompensas.",
        "",
        "· Cómo Funciona Conceptualmente: ",
        "El robot aprende qué acciones son buenas o malas basándose en experiencias pasadas.",
        "",
        "· Ejemplo: ",
        "Si el robot tomó una buena decisión, se guarda esa acción como 'buena'. Si fue mala, se evita en el futuro.",
        "",
        "· ¿Qué aporta el Deep Learning?: ",
        "DQN usa redes neuronales para manejar grandes entornos y estima las mejores acciones sin almacenar una tabla completa.",
        "",
        "• VENTAJAS",
        "· Adaptabilidad: ",
        "A diferencia de los métodos tradicionales, DQN permite que el robot se adapte a cambios imprevistos en el entorno.",
        "",
        "· Aprendizaje Continuo: ",
        "DQN aprende de cada acción que toma, incluso si es una mala decisión al principio.",
        "",
        "• LIMITACIONES",
        "· Entrenamiento Intensivo: ",
        "Requiere mucho tiempo y poder computacional para entrenar adecuadamente.",
        "",
        "· Exploración y Explotación: ",
        "Necesita explorar antes de explotar el conocimiento y tomar decisiones óptimas.",
        "",
        "3. TECNOLOGÍAS.",
        "• Python.",
        "• Pygame: Crear el entorno.",
        "• OpenAI GYM: Para entrenar el modelo.",
        "• TensorFlow: Para Deep Learning.",
        "• Deep Q Network: Para el modelo de DRL.",
        "",
        "Para volver dale a ESC."
    ]

    scroll_offset = 0
    line_height = 40
    ancho_pantalla = pantalla.get_width()

    fuente = crear_fuente(pantalla, porcentaje_altura=0.04)

    # Ajustar todas las líneas antes de empezar
    lineas_ajustadas = []
    for texto in informacion:
        lineas = ajustar_texto(texto, fuente, ancho_pantalla - 100)
        lineas_ajustadas.extend(lineas)

    # Calcular altura total real y scroll máximo
    total_altura = len(lineas_ajustadas) * line_height
    max_scroll = max(0, total_altura - pantalla.get_height() + 50)  # +50 de margen superior

    esperando = True
    while esperando:
        pantalla.fill((0, 0, 0))
        y_offset = 50 - scroll_offset

        for linea in lineas_ajustadas:
            if y_offset + line_height > 0 and y_offset < pantalla.get_height():
                texto_renderizado = fuente.render(linea, True, (255, 255, 255))
                pantalla.blit(texto_renderizado, (50, y_offset))
            y_offset += line_height

        pygame.display.flip()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    esperando = False
                elif evento.key == pygame.K_DOWN:
                    scroll_offset = min(scroll_offset + line_height, max_scroll)
                elif evento.key == pygame.K_UP:
                    scroll_offset = max(scroll_offset - line_height, 0)