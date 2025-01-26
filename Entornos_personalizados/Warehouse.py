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
        'render_modes': ['human'],
        'render_fps': 30
    }

    def __init__(self):
        # Inicialización de Pygame
        pygame.init()

        # Configuración del entorno
        self.current_score = 0  # Puntuación acumulada del agente
        self.current_episode = 0  # Número de episodios jugados
        self.state = np.array([5.0])  # Estado inicial del agente
        self.current_step = 0  # Contador de pasos
        self.done = False  # Indicador de finalización del episodio
        self.max_steps = 100  # Máximo número de pasos por episodio

        # Definir la cuenta regresiva y el tiempo inicial (en segundos)
        self.countdown_time = 60  # Duración máxima del episodio

        # Espacios de observación y acción
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Dos acciones: 0 (izquierda) o 1 (derecha)

        # Crear la ventana de Pygame
        self.screen = pygame.display.set_mode((1280, 820))
        pygame.display.set_caption("Warehouse Environment")

        # Cargar imagen de fondo
        self.background_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\fondo.png").convert()

        # Dirección inicial del agente
        self.agent_angle = 0  # Ángulo de orientación inicial
        self.target_angle = 0  # Ángulo objetivo para rotaciones
        self.agent_speed = 0.5  # Velocidad de movimiento del agente
        self.rotation_speed = 5  # Velocidad de rotación del agente

        # Inicializar la posición del agente
        self.agent_position = [640, 410]  # Posición inicial en el centro de la ventana

        # Cargar y redimensionar la imagen del agente
        self.agent_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\agente.png").convert_alpha()
        original_width, original_height = self.agent_image.get_size()
        scale_factor = 0.2  # Factor de escalado
        self.agent_width = int(original_width * scale_factor)
        self.agent_height = int(original_height * scale_factor)
        self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))

        # Cargar la imagen del premio
        self.reward_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\reward.png").convert_alpha()

        # Definir la posición y el tamaño del cuadrado de recompensa
        self.reward_square_size = 30  # Tamaño del cuadrado de recompensa
        self.reward_position = [500, 300]  # Posición inicial del premio

        self.agent_mask = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\mascaraAgente.png").convert_alpha()
        self.environment_mask = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\mascaraFondo.png").convert()

        self.agent_mask = pygame.transform.scale(self.agent_mask, (self.agent_width, self.agent_height))

        # Inicializar el tiempo de inicio del episodio
        self.start_time = time.time()  # Registrar el tiempo inicial

        # Variable de control para el estado de la ventana
        self.window_open = True  # Indica si la ventana de Pygame está abierta

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.

        Returns:
            state (np.array): Estado inicial del agente.
            info (dict): Información adicional (vacío en este caso).
        """
        if seed is not None:
            np.random.seed(seed)

        # Restablecer los parámetros iniciales
        self.state = np.array([5.0])  # Resetear el estado
        self.done = False
        self.current_step = 0
        self.current_score = 0  # Reiniciar la puntuación
        self.current_episode += 1  # Incrementar el episodio
        self.countdown_time = 60
        self.start_time = time.time()  # Reiniciar el tiempo de inicio al reiniciar el entorno
        self.reward_position = self.get_random_reward_position()  # Nueva posición de recompensa
        self.reward_collected = False  # Reiniciar la bandera de recompensa recogida
        # Devuelve la observación (estado) como float32 y un diccionario vacío para la nueva API.
        
        # DEBUG: Verifica la posición de la recompensa
        print(f"Reiniciando episodio {self.current_episode}. Nueva posición de recompensa: {self.reward_position}")

        self.agent_position = [170, 120]  # Volver a la posicion inicial

        # DEBUG: Verifica la posición inicial del agente
        print(f"Posición inicial del agente: {self.agent_position}")

        return np.array(self.agent_position, dtype=np.float32), {}

    def check_collision(self):
        """
        Verifica si las zonas blancas del agente y el entorno se superponen.

        Si la zona blanca del agente toca la zona blanca del entorno, reinicia el entorno.
        """
        # Crear las máscaras del agente y del entorno usando las imágenes cargadas
        agent_mask = pygame.mask.from_surface(self.agent_mask)
        environment_mask = pygame.mask.from_surface(self.environment_mask)

        # Calcular el desplazamiento del agente para que las máscaras se alineen correctamente
        offset = (self.agent_position[0], self.agent_position[1])
        print("offset: ", offset)
        # Verificar la superposición de las zonas blancas
        overlap = agent_mask.overlap(environment_mask, offset)
        print("overlap: ", overlap)
        if overlap:  # Si hay una superposición
            print("Colisión detectada, reiniciando entorno...")
            return True
        return False
    def get_random_reward_position(self):
        """
        Genera una posición aleatoria dentro de los límites de la pantalla para la recompensa.

        Este método garantiza que el premio aparezca dentro de los límites visibles de la pantalla y 
        en un lugar diferente cada vez que se llama.

        Returns:
            tuple: Coordenadas (x, y) de la nueva posición del premio.
        """
        x = np.random.randint(100, 1180 - self.reward_square_size)
        y = np.random.randint(100, 680 - self.reward_square_size)
        return (x, y)

    def step(self, action):
        """
        Realiza un paso en el entorno basado en la acción tomada por el agente.

        Args:
            action (int): Acción elegida por el agente. Puede tomar los valores:
                - 0: Mover hacia la izquierda.
                - 1: Mover hacia la derecha.
                - 2: Mover hacia arriba.
                - 3: Mover hacia abajo.

        Returns:
            state (np.array): Nuevo estado del entorno.
            reward (float): Recompensa obtenida tras realizar la acción.
            done (bool): Indicador de si el episodio ha terminado.
            truncated (bool): Indicador de si el episodio fue truncado.
            info (dict): Información adicional.
        """
        self.current_step += 1  # Incrementar el contador de pasos
        # DEBUG: Verifica el tiempo restante
        print(f"Pasos: {self.current_step}")

        # Resta el tiempo de la cuenta regresiva (basado en tiempo real)
        elapsed_time = time.time() - self.start_time
        self.countdown_time = max(0, 60 - int(elapsed_time))  # Tiempo restante en segundos

        # DEBUG: Verifica el tiempo restante
        print(f"Tiempo restante: {self.countdown_time} segundos")

        # Explicación: La cuenta regresiva se reduce con el tiempo transcurrido desde que comenzó el episodio.
        # Si el tiempo alcanza cero, el episodio termina automáticamente.

        # Actualizar el estado basado en la acción
        if action == 0:  # Movimiento hacia la izquierda
            self.state -= 1
            self.agent_position[0] -= self.agent_speed # Mover el agente a la izquierda 
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

        # DEBUG: Verifica la nueva posición del agente después de la acción
        print(f"Posición del agente después de la acción: {self.agent_position}")

        # Rotar el agente lentamente hacia el ángulo objetivo
        angle_diff = (self.target_angle - self.agent_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        self.agent_angle = (self.agent_angle + self.rotation_speed * np.sign(angle_diff)) % 360

        # Si el tiempo se acabó, penalizar y reiniciar
        if self.countdown_time == 0:
            reward = -10  # Penalización cuando el tiempo se acaba
            self.done = True  # Marcar el episodio como terminado
            print(f"Tiempo agotado. Episodio terminado. Recompensa: {reward}")
            return self.state.astype(np.float32), reward, self.done, True, {}
        
        self.current_step += 1  # Incrementar el contador de pasos

        # Comprobar si hay colisión entre las zonas blancas
        if self.check_collision():
            state, _ = self.reset()
            return state, -10, True, False, {}  # Penalización por colisión, reiniciando el entorno

        # Verificar si el agente toca el cuadrado de recompensa
        agent_rect = pygame.Rect(self.agent_position[0], self.agent_position[1], self.agent_width, self.agent_height)
        reward_rect = pygame.Rect(self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size)
        
        # DEBUG: Verifica las posiciones de colisión del agente y la recompensa
        print(f"Posición del agente: {self.agent_position}")
        print(f"Posición de la recompensa: {self.reward_position}")
        
        # Comprobar si hay colisión entre el agente y el cuadrado de recompensa
        if agent_rect.colliderect(reward_rect) and not self.reward_collected:
            # El agente recoge la recompensa, aumentando su puntuación y generando un nuevo premio.
            reward = 10  # Recompensa por recoger el premio
            self.reward_position = self.get_random_reward_position()  # Mover el premio a una nueva posición
            self.reward_collected = True  # Marcar la recompensa como recogida
        else:
            reward = 0  # Sin recompensa

        # Limitar el estado dentro de los valores válidos
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Verificar si el episodio ha terminado
        if self.countdown_time == 0:
            self.done = True
            truncated = False  # No se truncó el episodio, solo se terminó
            print(f"Episodio terminado después de {self.current_step} pasos.")

        else:
            self.done = False
            truncated = False  # El episodio no se truncó

        return self.state.astype(np.float32), reward, self.done, truncated, {}

    def render(self, mode="human"):
        """
        Renderiza el entorno en la ventana de Pygame.

        Este método dibuja el fondo, al agente, y la recompensa en sus posiciones actuales. También 
        muestra información sobre el episodio, como la puntuación acumulada, el número de episodios, 
        y el tiempo restante. Actualmente, solo soporta el modo 'human'.

        Args:
            mode (str): Modo de renderizado (solo se admite 'human').
        """
        if mode == 'human':
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.window_open = False
                    pygame.quit()
                    return

            # Dibujar el fondo y el agente
            self.screen.fill((169, 169, 169))
            self.screen.blit(self.background_image, (20, 20))
            #self.screen.blit(self.environment_mask, (20, 20)) # comprobar la mascara

            # Rotar y dibujar el agente
            rotated_agent = pygame.transform.rotate(self.agent_image, -self.agent_angle)
            #rotated_agent = pygame.transform.rotate(self.agent_mask, -self.agent_angle) # comprobar la mascara
            agent_rect = rotated_agent.get_rect(center=(self.agent_position[0], self.agent_position[1]))
            self.screen.blit(rotated_agent, agent_rect.topleft)

            # Dibujar la recompensa
            self.screen.blit(self.reward_image, (self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size))

            # Mostrar información en pantalla
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Puntuación: {self.current_score}", True, (0, 0, 0))
            self.screen.blit(score_text, (40, 680))

            episode_text = font.render(f"Episodio: {self.current_episode}", True, (0, 0, 0))
            self.screen.blit(episode_text, (40, 720))

            countdown_text = font.render(f"Tiempo restante: {self.countdown_time}", True, (0, 0, 0))
            self.screen.blit(countdown_text, (25, 25))

            pygame.display.flip()

    def close(self):
        """
        Cierra el entorno y Pygame.

        Este método imprime información final sobre el último episodio antes de cerrar la ventana.
        Por ejemplo, si has abierto archivos, conexiones de red o cualquier otro recurso externo,
        debes asegurarte de liberarlos aquí para evitar fugas de memoria o problemas de rendimiento.
        """
        print(f"Último episodio: {self.current_episode}, Puntuación final: {self.current_score}")
        self.window_open = False
        pygame.quit()

# Ejemplo de uso (en RegistrarEntorno.py está puesto así)
# env = MiEntorno()
# state, _ = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Acción aleatoria
#     state, reward, done, truncated, info = env.step(action)
#     env.render()
# env.close()
