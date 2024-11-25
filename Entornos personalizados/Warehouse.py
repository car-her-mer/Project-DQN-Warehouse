# Este archivo define un entorno personalizado para OpenAI Gym. 
# Podrías utilizar este archivo para crear un entorno único que interactúe con Pygame para la visualización.

import gym
from gym import spaces
import numpy as np
import pygame

class MiEntorno(gym.Env):
    """
    Un entorno personalizado para OpenAI Gym con visualización en Pygame.
    """
    metadata = {
        'render_modes': ['human'],  # Declarar correctamente el modo de renderizado
        'render_fps': 30            # Establecer la tasa de fotogramas por segundo (FPS)
    }

    def __init__(self):
        # Inicialización de Pygame
        pygame.init()

        # Inicialización del entorno
        self.current_score = 0
        self.current_episode = 0
        self.state = np.array([5.0])  # Estado inicial
        self.current_step = 0
        self.done = False
        self.max_steps = 100  # Número máximo de pasos en un episodio

        # Definir los espacios de observación y acción
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Dos acciones: 0 (izquierda) o 1 (derecha)
        
        # Crear la ventana de Pygame
        self.screen = pygame.display.set_mode((1280, 820))  # Tamaño de la ventana
        pygame.display.set_caption("Warehouse Environment")

        # Dirección inicial del agente (en grados, 0 = derecha, 90 = abajo, 180 = izquierda, 270 = arriba)
        self.agent_angle = 0  
        self.target_angle = 0  # Ángulo objetivo para rotación suave
        # Velocidad del agente (en píxeles por paso)
        self.agent_speed = 0.5  # Movimiento más lento, como un robot con ruedas (mayor numero = mayor velocidad y viceversa)
        self.rotation_speed = 5  # Velocidad de rotación (en grados por frame)
        
        # Inicializar la posición del agente (x, y)
        self.agent_position = [640, 410]  # Centro de la pantalla

        # Cargar y redimensionar la imagen del agente
        self.agent_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\agente.png").convert_alpha()
        self.agent_width = 50  # Ancho del agente
        self.agent_height = 50  # Alto del agente (si es necesario)
        self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))  # Redimensionar la imagen

        # Variable de control para el estado de la ventana
        self.window_open = True

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        """
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([5.0])  # Resetear el estado
        self.done = False
        self.current_step = 0
        self.current_score = 0  # Reiniciar la puntuación
        self.current_episode += 1  # Incrementar el episodio
        # Devuelve la observación (estado) como float32 y un diccionario vacío para la nueva API.
        return self.state.astype(np.float32), {}

    def step(self, action):
        """
        Realiza un paso en el entorno.
        """
        self.current_step += 1  # Incrementar el contador de pasos

        # Actualizar el estado basado en la acción
        if action == 0:  # Movimiento hacia la izquierda
            self.state -= 1
            self.agent_position[0] -= self.agent_speed  # Mover el agente a la izquierda 
            self.target_angle = 180  # Establecer ángulo objetivo hacia la izquierda
        elif action == 1:  # Movimiento hacia la derecha
            self.state += 1
            self.agent_position[0] += self.agent_speed  # Mover el agente a la derecha (menor numero = menor velocidad)
            self.target_angle = 0  # Establecer ángulo objetivo hacia la derecha
        # (Opcional) Si incluyes movimientos arriba y abajo:
        elif action == 2:  # Movimiento hacia arriba
            self.state += 0.5
            self.agent_position[1] -= self.agent_speed
            self.target_angle = 270  # Establecer ángulo objetivo hacia arriba
        elif action == 3:  # Movimiento hacia abajo
            self.state -= 0.5
            self.agent_position[1] += self.agent_speed
            self.target_angle = 90  # Establecer ángulo objetivo hacia abajo

         # Rotar el agente lentamente hacia el ángulo objetivo
        angle_diff = (self.target_angle - self.agent_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        rotation_step = min(self.rotation_speed, abs(angle_diff)) * np.sign(angle_diff)
        self.agent_angle = (self.agent_angle + rotation_step) % 360

        # Limitar la posición del agente para que no se salga de la pantalla
        self.agent_position[0] = max(0, min(self.agent_position[0], 1280 - self.agent_width))
        self.agent_position[1] = max(0, min(self.agent_position[1], 720 - self.agent_height))

        # Limitar el estado al rango permitido
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Verificar si el episodio ha terminado
        if self.current_step >= self.max_steps:
            self.done = True
            truncated = False  # No se truncó el episodio, solo se terminó
        else:
            self.done = False
            truncated = False  # El episodio no se truncó

        # Calcular la recompensa
        reward = 1.0 if self.state[0] == 10 else -0.1

        # Actualizar la puntuación (por ejemplo, acumulando recompensas)
        self.current_score += reward

        # Devolver la observación (convertida a float32), la recompensa, el estado de finalización y truncamiento, y un diccionario vacío
        return self.state.astype(np.float32), reward, self.done, truncated, {}

    def render(self, mode="human"):
        """
        Renderiza el entorno en la ventana de Pygame.
        """
        if mode == 'human':
            # Manejo de eventos de Pygame (como cerrar la ventana)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.window_open = False  # Marcar la ventana como cerrada
                    pygame.quit()
                    return

            # Rellenar la pantalla de gris
            self.screen.fill((169, 169, 169))  # Color gris (RGB)

            # Aquí puedes agregar tu código de renderizado, como dibujar elementos en la pantalla
            # Dibujar un segundo rectángulo más pequeño (blanco)
            # pygame.draw.rect(self.screen, (255, 255, 255), (20, 20, 1240, 645))  # x, y, ancho, alto

            # Cargar la imagen
            mi_dibujo = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\fondo.png")
            self.screen.blit(mi_dibujo, (20, 20))

            # Dibujar el agente con la rotación actual
            rotated_agent = pygame.transform.rotate(self.agent_image, self.agent_angle)
            agent_rect = rotated_agent.get_rect(center=(self.agent_position[0] + self.agent_width // 2,
                                                        self.agent_position[1] + self.agent_height // 2))
            self.screen.blit(rotated_agent, agent_rect.topleft)
            #print(f"Posición del agente: {self.agent_position}")

            # Mostrar el estado (para debug)
            font = pygame.font.Font(None, 36)
            # text = font.render(f"Estado: {self.state[0]}", True, (0, 0, 0))
            # self.screen.blit(text, (50, 50))  # Mostrar texto en la ventana

            # Mostrar la puntuación actual
            score_text = font.render(f"Puntuación: {self.current_score}", True, (0, 0, 0))
            self.screen.blit(score_text, (40, 680))  # Posición debajo del rectángulo blanco

            # Mostrar el episodio actual
            episode_text = font.render(f"Episodio: {self.current_episode}", True, (0, 0, 0))
            self.screen.blit(episode_text, (40, 720))  # Posición debajo de la puntuación

            # Actualizar la pantalla
            pygame.display.flip()

    def close(self):
        """
        Cierra el entorno y Pygame.
        """
        print(f"Último episodio: {self.current_episode}, Puntuación final: {self.current_score}")
        self.window_open = False
        pygame.quit()