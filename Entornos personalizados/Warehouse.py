# Este archivo define un entorno personalizado para OpenAI Gym. 
# Podrías utilizar este archivo para crear un entorno único que interactúe con Pygame para la visualización.

import gym
from gym import spaces
import numpy as np
import pygame
import time

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

        # Definir la cuenta regresiva y el tiempo inicial (en segundos o pasos)
        self.countdown_time = 60  # 1 minuto = 60 segundos  # Por ejemplo, 300 segundos (5 minutos)

        # Definir los espacios de observación y acción
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Dos acciones: 0 (izquierda) o 1 (derecha)
        
        # Crear la ventana de Pygame
        self.screen = pygame.display.set_mode((1280, 820))  # Tamaño de la ventana
        pygame.display.set_caption("Warehouse Environment")

        self.background_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\fondo.png").convert()

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
        #self.agent_width = 50  # Ancho del agente
        #self.agent_height = 50  # Alto del agente (si es necesario)
        #self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))  # Redimensionar la imagen

         # Máscara de colisiones (generada previamente)
        self.collision_mask = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\mascaraFondo.png").convert()
        self.collision_mask = pygame.surfarray.array_alpha(self.collision_mask) == 0  # Zonas negras = obstáculos

        original_width, original_height = self.agent_image.get_size()
        scale_factor = 0.2  # Ajusta este valor para el tamaño final deseado
        self.agent_width = int(original_width * scale_factor)
        self.agent_height = int(original_height * scale_factor)
        self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))

        # Definir la posición y el tamaño del cuadrado de recompensa
        self.reward_square_size = 30  # Tamaño del cuadrado de recompensa
        self.reward_position = [500, 300]  # Posición en el espacio (x, y)

        # Inicializar el tiempo de inicio del episodio
        self.start_time = time.time()

        # Variable de control para el estado de la ventana
        self.window_open = True

    def check_collision(self, x, y):
        #Verifica si el agente colisiona con una zona no permitida.
        agent_rect = pygame.Rect(x, y, self.agent_width, self.agent_height)
        for px in range(max(0, agent_rect.left), min(self.collision_mask.shape[1], agent_rect.right)):
            for py in range(max(0, agent_rect.top), min(self.collision_mask.shape[0], agent_rect.bottom)):
                if self.collision_mask[py, px]:  # Verificar si la posición es colisión
                    return True
        return False
    
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
        self.countdown_time = 60  # Resetear el tiempo de cuenta regresiva
        self.reward_position = self.get_random_reward_position()  # Nueva posición de recompensa
        # Devuelve la observación (estado) como float32 y un diccionario vacío para la nueva API.
        self.agent_position = [640, 410]  # Volver al centro
        return np.array(self.agent_position, dtype=np.float32), {}

    def get_random_reward_position(self):
        # Genera una posición aleatoria dentro de los límites del fondo para la recompensa.
        # Limitar el área dentro de los bordes permitidos del fondo
        while True:
            x = np.random.randint(100, 1180 - self.reward_square_size)
            y = np.random.randint(100, 680 - self.reward_square_size)
            self.reward_rect = pygame.Rect(x, y, self.reward_square_size, self.reward_square_size)

            # Verificar si la posición de la recompensa está en una zona permitida
            reward_area = self.collision_mask[y:y+self.reward_square_size, x:x+self.reward_square_size]

            # Verificar si toda el área de la recompensa está dentro de las zonas blancas (permitidas)
            # Usamos `== 255` porque el blanco es el valor permitido (si la máscara es en blanco y negro)
            if np.all(reward_area == 255):  # El valor 255 en la máscara indica una zona libre
                return (x, y)
            
    def step(self, action):
        """
        Realiza un paso en el entorno.
        """
        self.current_step += 1  # Incrementar el contador de pasos

        # Resta el tiempo de la cuenta regresiva (basado en tiempo real)
        elapsed_time = time.time() - self.start_time
        self.countdown_time = max(0, 60 - int(elapsed_time))  # Tiempo restante en segundos

        # Si el tiempo se acabó, penalizar y reiniciar
        if self.countdown_time == 0:
            reward = -10  # Penalización cuando el tiempo se acaba
            self.reset()  # Reiniciar el episodio
            return self.state.astype(np.float32), reward, self.done, False, {}

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

        # Verificar colisión antes de mover
        if not self.check_collision(self.agent_position[0], self.agent_position[1]):
            self.agent_position = [self.agent_position[0], self.agent_position[1]]

        # Rotar el agente lentamente hacia el ángulo objetivo
        angle_diff = (self.target_angle - self.agent_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        rotation_step = min(self.rotation_speed, abs(angle_diff)) * np.sign(angle_diff)
        self.agent_angle = (self.agent_angle + rotation_step) % 360

        # Verificar si el agente toca el cuadrado de recompensa
        agent_rect = pygame.Rect(self.agent_position[0], self.agent_position[1], self.agent_width, self.agent_height)
        reward_rect = pygame.Rect(self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size)

        # Comprobar si hay colisión entre el agente y el cuadrado de recompensa
        if agent_rect.colliderect(reward_rect):
            reward = 10  # Sumar una recompensa cuando el agente toca el cuadrado
            self.current_score += reward  # Incrementar la puntuación
            self.reward_position = self.get_random_reward_position()  # Reubicar la recompensa
            self.current_score = 0  # Reiniciar la puntuación cuando toque la recompensa
        else:
            reward = -0.1  # Penalización si no toca el cuadrado

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
        #reward = 1.0 if self.state[0] == 10 else -0.1

        # Actualizar la puntuación (por ejemplo, acumulando recompensas)
        #self.current_score += reward

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
            self.screen.blit(self.background_image, (20, 20))
            
            # Dibujar el agente con rotación y punto de giro ajustado
            rotated_agent = pygame.transform.rotate(self.agent_image, -self.agent_angle)  # Rotación antihoraria
            # Ajustar el centro del rectángulo del agente para ser el punto de giro
            agent_rect = rotated_agent.get_rect(center=(self.agent_position[0], self.agent_position[1]))

            self.screen.blit(rotated_agent, agent_rect.topleft)
            #print(f"Posición del agente: {self.agent_position}")

            # Dibujar el cuadrado de recompensa
            pygame.draw.rect(self.screen, (0, 0, 0), (self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size))

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

            # Mostrar la cuenta regresiva
            countdown_text = font.render(f"Tiempo restante: {self.countdown_time}", True, (0, 0, 0))
            self.screen.blit(countdown_text, (10, 10))  # Posición cerca de la puntuación

            # Actualizar la pantalla
            pygame.display.flip()

    def close(self):
        """
        Cierra el entorno y Pygame.
        """
        print(f"Último episodio: {self.current_episode}, Puntuación final: {self.current_score}")
        self.window_open = False
        pygame.quit()