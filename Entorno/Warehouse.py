# Entorno personalizado para OpenAI Gym con visualización en Pygame.
# Este archivo define la clase MiEntorno, que simula un almacén donde un agente (robot)
# puede moverse, recoger recompensas y aprender usando aprendizaje por refuerzo.
# Aquí se gestionan la lógica del entorno, la visualización y la interacción con el agente.

import gym  # Librería para entornos de aprendizaje por refuerzo
from gym import spaces  # Para definir espacios de observación y acción
import numpy as np  # Para operaciones matemáticas y matrices
import pygame  # Para la visualización gráfica
import time  # Para controlar el tiempo

class MiEntorno(gym.Env):
    """
    Un entorno personalizado para OpenAI Gym con visualización en Pygame.
    """
    metadata = {
        'render_modes': ['human'], # Declarar correctamente el modo de renderizado
        'render_fps': 30 # Establecer la tasa de fotogramas por segundo (FPS)
    }

    def __init__(self):
        # Inicialización de Pygame (necesario para usar gráficos)
        pygame.init()

        # Variables para llevar el control de la puntuación y episodios
        self.current_score = 0  # Puntuación acumulada del agente
        self.best_score = 0  # Mejor puntuación alcanzada
        self.current_episode = 0  # Número de episodios jugados
        self.best_episode = 0  # Episodio con mejor puntuación
        self.reward = 0  # Recompensa actual
        self.current_reward = 0  # Número de recompensas recogidas en el episodio
        self.best_reward = 0  # Mejor número de recompensas recogidas

        # self.state = np.array([5.0])  # Estado inicial del agente (comentado)
        self.current_step = 0  # Contador de pasos en el episodio
        self.done = False  # Indica si el episodio terminó
        self.max_steps = 100  # Máximo número de pasos por episodio

        # Definir la cuenta regresiva y el tiempo inicial (en segundos)
        self.sec = 60  # Duración máxima del episodio en segundos
        self.countdown_time = self.sec  # Tiempo restante

        # Espacios de observación y acción
        # self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1280, 820]), dtype=np.float32)  # Estado: posición (x, y)
        self.action_space = spaces.Discrete(4)  # Cuatro acciones: izquierda, derecha, arriba, abajo

        # Crear la ventana de Pygame para mostrar el entorno
        self.screen = pygame.display.set_mode((1280, 820))
        pygame.display.set_caption("Warehouse Environment")

        # Cargar imagen de fondo
        self.background_image = pygame.image.load("Assets\\fondo.png").convert()

        # Dirección y velocidad del agente
        self.agent_angle = 0  # Ángulo de orientación inicial
        self.target_angle = 0  # Ángulo objetivo para rotaciones
        self.agent_speed = 0.5  # Velocidad de movimiento del agente
        self.rotation_speed = 5  # Velocidad de rotación del agente

        # Inicializar la posición del agente y entorno
        self.eje1 = 170  # Posición inicial eje X
        self.eje2 = 120  # Posición inicial eje Y
        self.agent_position = np.array([self.eje1, self.eje2])  # Posición inicial del agente
        self.state = self.agent_position  # Estado inicial
        self.environment_position = (0, 0)  # El entorno está fijo en la posición (0, 0)

        # Cargar y redimensionar la imagen del agente
        self.agent_image = pygame.image.load("Assets\\agente.png").convert_alpha()
        original_width, original_height = self.agent_image.get_size()
        scale_factor = 0.2  # Factor de escalado para hacer el agente más pequeño
        self.agent_width = int(original_width * scale_factor)
        self.agent_height = int(original_height * scale_factor)
        self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))

        # Cargar la imagen del premio (recompensa)
        self.reward_image = pygame.image.load("Assets\\reward.png").convert_alpha()

        # Definir la posición y el tamaño del cuadrado de recompensa
        self.reward_square_size = 30  # Tamaño del cuadrado de recompensa
        self.reward_position = [500, 300]  # Posición inicial del premio

        # Cargar máscaras para colisiones
        self.agent_mask_image = pygame.image.load("Assets\\mascaraAgente.png").convert_alpha()
        self.environment_mask = pygame.image.load("Assets\\mascaraFondoAlpha.png").convert_alpha()
        self.agent_mask_image = pygame.transform.scale(self.agent_mask_image, (self.agent_width, self.agent_height))
        self.agent_mask = pygame.mask.from_surface(self.agent_mask_image)  # Máscara del agente
        if not hasattr(self, 'environment_mask_obj'):
            self.environment_mask_obj = pygame.mask.from_surface(self.environment_mask)  # Máscara del fondo

        # Inicializar el tiempo de inicio del episodio
        self.start_time = time.time()  # Guardar el tiempo inicial

        # Variable para saber si la ventana está abierta
        self.window_open = True  # Indica si la ventana de Pygame está abierta

    def reset(self, seed=None, options=None):
        # Reiniciar el entorno para un nuevo episodio
        if seed is not None:
            np.random.seed(seed)

        # Restablecer los parámetros iniciales
        self.agent_position = np.array([self.eje1, self.eje2], dtype=np.float64)  # Posición inicial
        self.state = np.copy(np.array(self.agent_position, dtype=np.float64))  # Copia del estado
        self.done = False
        self.current_step = 0
        self.current_score = 0  # Reiniciar la puntuación
        self.current_reward = 0
        self.reward = 0
        self.countdown_time = self.sec
        self.start_time = time.time()  # Reiniciar el tiempo de inicio
        self.reward_position = self.get_random_reward_position()  # Nueva posición de recompensa
        self.reward_collected = False  # Bandera de recompensa recogida
        self.episode_restarted = False  # El episodio acaba de reiniciarse
        print("posicion inicial en del reset:", self.agent_position)  # Mostrar posición inicial
        return self.state

    def check_collision(self):
        """
        Verifica si las zonas blancas del agente y el entorno se superponen.
        Si la zona blanca del agente toca la zona blanca del entorno, reinicia el entorno.
        """
        # Calcular el desplazamiento relativo entre las posiciones
        offset = (
            self.agent_position[0] - self.environment_position[0],
            self.agent_position[1] - self.environment_position[1]
        )
        # Verificar si hay superposición entre las máscaras
        collision = self.environment_mask_obj.overlap(self.agent_mask, offset)
        if collision:
            print(f"Colisión detectada en {collision}")
        return collision is not None

    def get_random_reward_position(self):
        # Genera una posición aleatoria para la recompensa dentro de la zona útil
        x = np.random.randint(150, 1128 - self.reward_square_size)
        y = np.random.randint(89, 598 - self.reward_square_size)
        return (x, y)

    def step(self, action):
        # Ejecuta una acción y actualiza el entorno
        prev_position = self.agent_position.copy()
        # Actualizar el tiempo restante
        elapsed_time = time.time() - self.start_time
        self.countdown_time = max(0, self.sec - int(elapsed_time))
        # Calcular la distancia a la recompensa antes de moverse
        reward_direction_x = self.reward_position[0] - self.agent_position[0]
        reward_direction_y = self.reward_position[1] - self.agent_position[1]
        distance_before = np.sqrt(reward_direction_x**2 + reward_direction_y**2)
        # Actualizar la posición del agente según la acción
        if not self.episode_restarted:
            self.current_step += 1
            if action == 0:  # Izquierda
                self.agent_position[0] -= self.agent_speed
                self.agent_position[0] = np.clip(self.agent_position[0], self.eje1, 1145)
                self.target_angle = 180
            elif action == 1:  # Derecha
                self.agent_position[0] += self.agent_speed
                self.agent_position[0] = np.clip(self.agent_position[0], self.eje1, 1145 - self.agent_width)
                self.target_angle = 0
            elif action == 2:  # Arriba
                self.agent_position[1] -= self.agent_speed
                self.agent_position[1] = np.clip(self.agent_position[1], self.eje2, 617)
                self.target_angle = 270
            elif action == 3:  # Abajo
                self.agent_position[1] += self.agent_speed
                self.agent_position[1] = np.clip(self.agent_position[1], self.eje2, 617 - self.agent_height)
                self.target_angle = 90
            print(f"Paso {self.current_step}: Acción {action}")
            print(f"    Posición antes: {prev_position}")
            print(f"    Posición después: {self.agent_position}")
            print(f"    Velocidad usada: {self.agent_speed}")
        else:
            self.current_step += 1
            self.episode_restarted = True
        self.state = np.array(self.agent_position, dtype=np.float64)
        # Rotar el agente hacia el ángulo objetivo
        angle_diff = (self.target_angle - self.agent_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        self.agent_angle = (self.agent_angle + self.rotation_speed * np.sign(angle_diff)) % 360
        # Si el tiempo se acabó o hay colisión, terminar episodio
        if self.countdown_time == 0 or self.check_collision():
            print("time out: ",self.reward)
            self.reward -= 10  # Penalización por tiempo agotado o colisión
            self.done = True  # Marcar el episodio como terminado
            self.current_score -= 10  # Restar puntos por perder
            truncated = True  # Indica que el episodio terminó antes de tiempo
            self.episode_restarted = True  # Señal para reiniciar el episodio
            # Si hubo colisión, mostrar mensaje especial
            if self.check_collision():
                print("¡Colisión detectada! Reiniciando episodio.")
            else:
                print(f"Tiempo agotado. Episodio terminado. Recompensa: {self.reward}")
                print(f"Current score: {self.current_score}")
            # Si el número de recompensas de este episodio es el mejor, actualizar récords
            if self.current_reward > self.best_reward:
                self.best_score = self.current_score  # Guardar la mejor puntuación
                self.best_episode = self.current_episode  # Guardar el mejor episodio
                self.best_reward = self.current_reward  # Guardar el mejor número de recompensas
            self.current_episode += 1  # Pasar al siguiente episodio
            # Limitar el estado a los valores válidos del entorno
            self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
            # Devolver el estado, recompensa, si terminó, si fue truncado y un diccionario vacío (info extra)
            return self.state, self.reward, self.done, truncated, {}
        else:
            self.done = False  # El episodio sigue
            truncated = False  # No fue truncado
            self.episode_restarted = False  # No hay reinicio
            # Calcular la distancia después de mover al agente
            reward_direction_x = self.reward_position[0] - self.agent_position[0]
            reward_direction_y = self.reward_position[1] - self.agent_position[1]
            distance_after = np.sqrt(reward_direction_x**2 + reward_direction_y**2)
            # Evaluar si el agente se acerca o se aleja de la recompensa
            if distance_after < distance_before:
                self.reward += 1  # Se acerca al premio
                self.current_score += 1
            elif distance_after > distance_before:
                self.reward -= 1  # Se aleja del premio
                self.current_score -= 1
            # Verificar si el agente toca el cuadrado de recompensa
            reward_rect = pygame.Rect(self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size)
            agent_rect = pygame.Rect(
                self.agent_position[0] - self.agent_width // 2,
                self.agent_position[1] - self.agent_height // 2,
                self.agent_width,
                self.agent_height
            )
            if agent_rect.colliderect(reward_rect) and not self.reward_collected:
                self.reward += 50  # Recompensa por recoger el premio
                self.reward_position = self.get_random_reward_position()  # Nueva posición
                self.reward_collected = True
                self.current_score += self.reward
                self.current_reward += 1
            else:
                if not agent_rect.colliderect(reward_rect):
                    self.reward_collected = False
            self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
            print(f"estado en el state de reward: {self.reward}")
            self.render()
            return self.state, self.reward, self.done, truncated, {}

    def render(self, mode="human"):
        # Dibuja el entorno y la información en la ventana de Pygame
        if mode == 'human':
            if not self.window_open:
                self.close()
                return
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.window_open = False
                    self.close()
                    return
            
            # Dibujar el fondo y el agente
            self.screen.fill((169, 169, 169))  # Fondo gris
            self.screen.blit(self.background_image, (20, 20))
            
            # Rotar la imagen del agente según el ángulo actual (para simular dirección)
            rotated_agent = pygame.transform.rotate(self.agent_image, -self.agent_angle)
            
            # Obtener el rectángulo que rodea al agente para posicionarlo correctamente
            agent_rect = rotated_agent.get_rect(center=(self.agent_position[0], self.agent_position[1]))
            
            # Dibujar el agente rotado en la pantalla
            self.screen.blit(rotated_agent, agent_rect.topleft)
            
            # Rotar y dibujar la máscara del agente (opcional, útil para depuración de colisiones)
            rotated_mask_surface = pygame.transform.rotate(self.agent_mask_image, -self.agent_angle)
            mask_rect = rotated_mask_surface.get_rect(center=(self.agent_position[0], self.agent_position[1]))
            #self.screen.blit(rotated_mask_surface, mask_rect.topleft)  # Descomenta para ver la máscara
            
            # Dibujar la imagen de la recompensa en la posición actual
            self.screen.blit(self.reward_image, (self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size))
           
            # Mostrar información en pantalla (texto)
            font = pygame.font.Font(None, 36)  # Fuente para los textos
            
            # Crear y dibujar cada texto informativo en la pantalla
            countdown_text = font.render(f"Tiempo restante: {self.countdown_time}", True, (0, 0, 0))
            self.screen.blit(countdown_text, (25, 25))  # Tiempo restante arriba a la izquierda
            
            episode_text = font.render(f"Episodio: {self.current_episode}", True, (0, 0, 0))
            self.screen.blit(episode_text, (40, 680))  # Episodio actual
            
            best_episode_text = font.render(f"Mejor episodio: {self.best_episode}", True, (0, 0, 0))
            self.screen.blit(best_episode_text, (350, 680))  # Mejor episodio
            
            reward_text = font.render(f"Núm. recompensas: {self.current_reward}", True, (0, 0, 0))
            self.screen.blit(reward_text, (40, 720))  # Recompensas recogidas en el episodio
            
            best_reward_text = font.render(f"Mejor núm. recompensas: {self.best_reward}", True, (0, 0, 0))
            self.screen.blit(best_reward_text, (350, 720))  # Mejor número de recompensas
            
            score_text = font.render(f"Puntuación: {self.current_score}", True, (0, 0, 0))
            self.screen.blit(score_text, (40, 760))  # Puntuación actual
            
            best_score_text = font.render(f"Mejor puntuación: {self.best_score}", True, (0, 0, 0))
            self.screen.blit(best_score_text, (350, 760))  # Mejor puntuación
            
            pygame.display.flip()  # Actualizar la pantalla para mostrar los cambios

    def close(self):
        # Cierra la ventana y termina Pygame
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
