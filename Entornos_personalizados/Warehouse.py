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
        'render_modes': ['human'], # Declarar correctamente el modo de renderizado
        'render_fps': 30 # Establecer la tasa de fotogramas por segundo (FPS)
    }

    def __init__(self):
        # Inicialización de Pygame
        pygame.init()

        # Configuración del entorno
        self.current_score = 0  # Puntuación acumulada del agente
        self.best_score = 0  # Variable para la mejor puntuación
        self.current_episode = 0  # Número de episodios jugados
        self.best_episode = 0  # Variable para el mejor episodio
        self.current_reward = 0
        self.best_reward = 0

        # self.state = np.array([5.0])  # Estado inicial del agente
        self.current_step = 0  # Contador de pasos
        self.done = False  # Indicador de finalización del episodio
        self.max_steps = 100  # Máximo número de pasos por episodio

        # Definir la cuenta regresiva y el tiempo inicial (en segundos)
        self.sec = 10
        self.countdown_time = self.sec  # Duración máxima del episodio

        # Espacios de observación y acción
        # self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1280, 820]), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # Cuatro acciones acciones: (izquierda, derecha, arriba, abajo)

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

        # Inicializar la posición del agente y entorno
        self.eje1 = 150
        self.eje2 = 120
        self.agent_position = [self.eje1, self.eje2]  # Posición inicial del agente (puede cambiar dinámicamente)
        #self.state = np.array([640.0, 410.0])  # Un valor dentro de los límites [0,0] y [1280,820]
        self.state = self.agent_position  # Un valor dentro de los límites [0,0] y [1280,820]
        #print(f"Posición del agente al init: {self.agent_position}")
        self.environment_position = (0, 0)  # El entorno está fijo en la posición (0, 0)

        # Cargar y redimensionar la imagen del agente
        self.agent_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\agente.png").convert_alpha()
        original_width, original_height = self.agent_image.get_size()
        scale_factor = 0.2  # Factor de escalado
        self.agent_width = int(original_width * scale_factor) #57
        self.agent_height = int(original_height * scale_factor) #67
        self.agent_image = pygame.transform.scale(self.agent_image, (self.agent_width, self.agent_height))

        # Cargar la imagen del premio
        self.reward_image = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\reward.png").convert_alpha()

        # Definir la posición y el tamaño del cuadrado de recompensa
        self.reward_square_size = 30  # Tamaño del cuadrado de recompensa
        self.reward_position = [500, 300]  # Posición inicial del premio

        self.agent_mask = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\mascaraAgente.png").convert_alpha()
        self.environment_mask = pygame.image.load("IA\\Project_RL-Warehouse\\Assets\\mascaraFondo.png").convert()

        self.agent_mask = pygame.transform.scale(self.agent_mask, (self.agent_width, self.agent_height))

        # Crear máscaras del agente y del entorno al inicio
        self.agent_mask = pygame.mask.from_surface(self.agent_mask)  # Máscara del agente
        if not hasattr(self, 'environment_mask_obj'):  # Crear solo una vez la máscara del fondo
            self.environment_mask_obj = pygame.mask.from_surface(self.environment_mask)

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
        # Restablecer la posición exacta del agente
        self.agent_position = np.array([self.eje1, self.eje2], dtype=np.float32)  # Garantizar tipo de dato correcto
        self.state = np.copy(np.array(self.agent_position, dtype=np.float32))  # Copia para evitar referencias inesperadas

        self.done = False
        self.current_step = 0
        self.current_score = 0  # Reiniciar la puntuación
        self.current_reward = 0
        self.countdown_time = self.sec
        self.start_time = time.time()  # Reiniciar el tiempo de inicio al reiniciar el entorno
        self.reward_position = self.get_random_reward_position()  # Nueva posición de recompensa
        self.reward_collected = False  # Reiniciar la bandera de recompensa recogida

        # Inicializar episodio reiniciado
        self.episode_restarted = False  # El episodio acaba de reiniciarse

        # DEBUG: Verifica la posición de la recompensa
        #print(f"Reiniciando episodio {self.current_episode}. Nueva posición de recompensa: {self.reward_position}")

        #print("Estado inicial después del reset:", self.state)  # Agregar impresión aquí
        return self.state

    def check_collision(self):
        """
        Verifica si las zonas blancas del agente y el entorno se superponen.

        Si la zona blanca del agente toca la zona blanca del entorno, reinicia el entorno.
        """
        #se quita de aquí para que no se genere la mascara todo el rato
        """# Crear máscaras del agente y del entorno
        agent_mask = pygame.mask.from_surface(self.agent_mask)  # Máscara del agente
        if not hasattr(self, 'environment_mask_obj'):  # Crear solo una vez la máscara del fondo
            self.environment_mask_obj = pygame.mask.from_surface(self.environment_mask)"""

        # Calcular el desplazamiento relativo entre las posiciones
        offset = (
            self.agent_position[0] - self.environment_position[0],
            self.agent_position[1] - self.environment_position[1]
        )

        # Verificar si hay superposición entre las máscaras
        collision = self.environment_mask_obj.overlap(self.agent_mask, offset)

        # DEBUG: Mostrar resultados de la colisión
        if collision:
            print(f"Colisión detectada en {collision}")
        """else:
            print("Sin colisión detectada.")"""

        #print(f"Posición del agente en collision: {self.agent_position}")

        return collision is not None

    def get_random_reward_position(self):
        """
        Genera una posición aleatoria dentro de los límites de la pantalla para la recompensa.

        Este método garantiza que el premio aparezca dentro de los límites visibles de la pantalla y 
        en un lugar diferente cada vez que se llama.

        Returns:
            tuple: Coordenadas (x, y) de la nueva posición del premio.
        """
        x = np.random.randint(150, 1128 - self.reward_square_size) #zona util
        y = np.random.randint(89, 598 - self.reward_square_size) #zona util
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
        prev_position = self.agent_position.copy()

        # Resta el tiempo de la cuenta regresiva (basado en tiempo real)
        elapsed_time = time.time() - self.start_time
        self.countdown_time = max(0, self.sec - int(elapsed_time))  # Tiempo restante en segundos

        # Obtener la posición relativa de la recompensa con respecto al agente
        reward_direction_x = self.reward_position[0] - self.agent_position[0]
        reward_direction_y = self.reward_position[1] - self.agent_position[1]

        # Calcular la distancia del agente a la recompensa antes de moverlo
        distance_before = np.sqrt(reward_direction_x**2 + reward_direction_y**2)

        # Actualizar el estado basado en la acción
        # np.clip(valor a limitar, valor minimo permitido y que no se mueva fuera de la zona util izquierda
            #, valor maximo permitido y que no se mueva fuera de la zona util derecha)
        # Actualizar el estado basado en la acción
        if not self.episode_restarted:
            self.current_step += 1  # Solo incrementar si el episodio no ha sido reiniciado
            # eje Y = agent_position[1] (arriba/abajo)
            # eje X = agent_position[0] (derechaizquierda)
            if action == 0:  # Movimiento hacia la izquierda
                #self.state -= 1
                self.agent_position[0] -= self.agent_speed  # Mover el agente a la izquierda
                # Asegurarse de que el borde izquierdo del agente no cruce el borde del entorno
                self.agent_position[0] = np.clip(self.agent_position[0], 150.5 + self.agent_width / 2, 1145)
                self.target_angle = 180  # Establecer ángulo objetivo hacia la izquierda
            elif action == 1:  # Movimiento hacia la derecha
                #self.state += 1
                self.agent_position[0] += self.agent_speed  # Mover el agente a la derecha
                # Asegurarse de que el borde derecho del agente no cruce el borde del entorno
                self.agent_position[0] = np.clip(self.agent_position[0], 150.5 + self.agent_width / 2, 1145 - self.agent_width)
                self.target_angle = 0  # Establecer ángulo objetivo hacia la derecha
            elif action == 2:  # Movimiento hacia arriba
                #self.state += 0.5
                self.agent_position[1] -= self.agent_speed  # Mover el agente hacia arriba
                # Asegurarse de que el borde superior del agente no cruce el borde del entorno
                self.agent_position[1] = np.clip(self.agent_position[1], 150.5 + self.agent_height / 2, 617)
                self.target_angle = 270  # Establecer ángulo objetivo hacia arriba
            elif action == 3:  # Movimiento hacia abajo
                #self.state -= 0.5
                self.agent_position[1] += self.agent_speed  # Mover el agente hacia abajo
                # Asegurarse de que el borde inferior del agente no cruce el borde del entorno
                self.agent_position[1] = np.clip(self.agent_position[1], 150.5 + self.agent_height / 2, 617 - self.agent_height)
                self.target_angle = 90  # Establecer ángulo objetivo hacia abajo
            #print(f"Posición del agente durante step: {self.agent_position}")
            print(f"Paso {self.current_step}: Acción {action}")
            print(f"    Posición antes: {prev_position}")
            print(f"    Posición después: {self.agent_position}")
            print(f"    Velocidad usada: {self.agent_speed}")

        else:
            self.current_step += 1  # Incrementar el contador de pasos
            self.episode_restarted = True
            #self.agent_position = np.array([self.eje1, self.eje2])  # Volver a la posicion inicial

        self.state = np.array(self.agent_position, dtype=np.float32)
        # Rotar el agente lentamente hacia el ángulo objetivo
        #print(f"Ángulo actual: {self.agent_angle}")
        #print(f"Ángulo objetivo: {self.target_angle}")
        angle_diff = (self.target_angle - self.agent_angle) % 360
        #print(f"Diferencia de ángulo (sin ajuste): {angle_diff}")
        if angle_diff > 180:
            angle_diff -= 360
        self.agent_angle = (self.agent_angle + self.rotation_speed * np.sign(angle_diff)) % 360

        #print(f"Diferencia de ángulo (ajustada): {angle_diff}")
        #print(f"Ángulo actual: {self.agent_angle} grados, Ángulo objetivo: {self.target_angle} grados")

        # Si el tiempo se acabó, penalizar y reiniciar
        if self.countdown_time == 0 or self.check_collision():
            reward = -10  # Penalización cuando el tiempo se acaba
            self.done = True  # Marcar el episodio como terminado
            self.current_score -= 10
            truncated = True
            self.episode_restarted = True  # El episodio ha terminado

            if self.check_collision():
                print("¡Colisión detectada! Reiniciando episodio.")
            else:
                print(f"Tiempo agotado. Episodio terminado. Recompensa: {reward}")

            # Al final del episodio, comparar la puntuación actual con la mejor puntuación
            if self.current_reward > self.best_reward:
                self.best_score = self.current_score  # Actualizar la mejor puntuación
                self.best_episode = self.current_episode  # Actualizar el mejor episodio
                self.best_reward = self.current_reward  # Actualizar el mejor núm. recompensas
                #print(f"¡Nueva mejor puntuación! {self.best_score}")
            """else:
                print(f"La mejor puntuación sigue siendo: {self.best_score}")"""
                
            # Aquí puedes verificar el estado después de aplicar la acción
            #print("Estado después de la acción:", self.state)

            if self.done:
                # Reiniciar el entorno para el siguiente episodio
                self.state, _ = self.reset()

            self.current_episode += 1    # Incrementar el episodio

            return self.state, reward, self.done, truncated, {}  
        else:
            self.done = False
            truncated = False  # El episodio no se truncó
            self.episode_restarted = False  # El episodio no ha terminado

            # Calcular la distancia después de mover al agente
            reward_direction_x = self.reward_position[0] - self.agent_position[0]
            reward_direction_y = self.reward_position[1] - self.agent_position[1]
            distance_after = np.sqrt(reward_direction_x**2 + reward_direction_y**2) 

            # Evaluar si el agente se acerca o se aleja de la recompensa
            if distance_after < distance_before:
                reward = 1  # El agente se acerca al premio
                self.current_score += 1
            elif distance_after > distance_before:
                reward = -1  # El agente se aleja del premio
                self.current_score -= 1
            else:
                reward = 0  # El agente no cambia su distancia al premio
            
            # Verificar si el agente toca el cuadrado de recompensa
            #agent_rect = pygame.Rect(self.agent_position[0], self.agent_position[1], self.agent_width, self.agent_height)
            reward_rect = pygame.Rect(self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size)
            agent_rect = pygame.Rect(
            self.agent_position[0] - self.agent_width // 2,
            self.agent_position[1] - self.agent_height // 2,
            self.agent_width,
            self.agent_height
            )

            # DEBUG: Verifica las posiciones de colisión del agente y la recompensa
            #print(f"Posición del agente: {self.agent_position}")
            #print(f"Posición de la recompensa: {self.reward_position}")
            
            # Comprobar si hay colisión entre el agente y el cuadrado de recompensa
            if agent_rect.colliderect(reward_rect) and not self.reward_collected:
                # El agente recoge la recompensa, aumentando su puntuación y generando un nuevo premio.
                reward = 50  # Recompensa por recoger el premio
                self.reward_position = self.get_random_reward_position()  # Mover el premio a una nueva posición
                self.reward_collected = True  # Marcar la recompensa como recogida
                self.current_score += reward  # Incrementar la puntuación del agente
                self.current_reward += 1
            else:
                # Si el agente ya ha recogido la recompensa, permitirlo nuevamente al salir de la zona
                if not agent_rect.colliderect(reward_rect):
                    self.reward_collected = False
                reward = -0.1  # Pequeña penalización para evitar quedarse quieto

            # Limitar el estado dentro de los valores válidos
            self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

            #print(f"estado en el state de step: {self.state}")
            
            return self.state, reward, self.done, truncated, {}

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
                    self.window_open = False # Marcar la ventana como cerrada
                    # pygame.quit()
                    return

            # Dibujar el fondo y el agente
            self.screen.fill((169, 169, 169))
            self.screen.blit(self.background_image, (20, 20))
            #self.screen.blit(self.environment_mask, (20, 20)) # comprobar la mascara

            # Rotar y dibujar el agente
            rotated_agent = pygame.transform.rotate(self.agent_image, -self.agent_angle)
            #rotated_agent = pygame.transform.rotate(self.agent_mask, -self.agent_angle)
            agent_rect = rotated_agent.get_rect(center=(self.agent_position[0], self.agent_position[1])) 
            
            self.screen.blit(rotated_agent, agent_rect.topleft)

            # Dibujar la recompensa
            self.screen.blit(self.reward_image, (self.reward_position[0], self.reward_position[1], self.reward_square_size, self.reward_square_size))

            # Mostrar información en pantalla
            font = pygame.font.Font(None, 36)

            countdown_text = font.render(f"Tiempo restante: {self.countdown_time}", True, (0, 0, 0))
            self.screen.blit(countdown_text, (25, 25))

            episode_text = font.render(f"Episodio: {self.current_episode}", True, (0, 0, 0))
            self.screen.blit(episode_text, (40, 680))

            best_episode_text = font.render(f"Mejor episodio: {self.best_episode}", True, (0, 0, 0))
            self.screen.blit(best_episode_text, (350, 680))

            reward_text = font.render(f"Núm. recompensas: {self.current_reward}", True, (0, 0, 0))
            self.screen.blit(reward_text, (40, 720))

            best_reward_text = font.render(f"Mejor núm. recompensas: {self.best_reward}", True, (0, 0, 0))
            self.screen.blit(best_reward_text, (350, 720))

            score_text = font.render(f"Puntuación: {self.current_score}", True, (0, 0, 0))
            self.screen.blit(score_text, (40, 760))

            best_score_text = font.render(f"Mejor puntuación: {self.best_score}", True, (0, 0, 0))
            self.screen.blit(best_score_text, (350, 760))

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
