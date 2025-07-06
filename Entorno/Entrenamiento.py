# Entrenamiento del agente con DQN en el entorno personalizado.
# Este archivo contiene la función principal que registra el entorno,
# define la red neuronal, ejecuta el bucle de entrenamiento y muestra el progreso.
# Aquí se gestiona el aprendizaje del agente y la visualización de resultados.

# Importar módulos necesarios para el funcionamiento del script
import sys
import os

# Añadir el directorio actual al sys.path para poder importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Intentar importar el entorno personalizado desde Warehouse.py
try:
    from Warehouse import MiEntorno
    print("Importación exitosa")
except ImportError as e:
    print(f"Error de importación: {e}")

# Importar librerías necesarias para el entorno y el aprendizaje
import gym  # Librería para entornos de refuerzo
import numpy as np  # Para operaciones numéricas
import tensorflow as tf  # Para redes neuronales
from keras import layers  # Para crear capas de la red
from gym.envs.registration import register  # Para registrar entornos personalizados
import matplotlib.pyplot as plt  # Importar Matplotlib
def IniciarEntorno():
    # Registrar el entorno personalizado en Gym
    register(
        id="MiEntorno-v1",  # Nombre único para el entorno
        entry_point="Warehouse:MiEntorno",  # Dónde encontrar la clase del entorno
    )

    # Crear una instancia del entorno
    env = gym.make("MiEntorno-v1")

    # --- DQN: Deep Q-Network ---
    # Parámetros del entorno
    state_size = env.observation_space.shape[0]  # Tamaño del vector de estado
    action_size = env.action_space.n  # Número de acciones posibles

    # Definir la red neuronal que usará el agente para aprender
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),  # Capa oculta 1
        layers.Dense(24, activation='relu'),  # Capa oculta 2
        layers.Dense(action_size, activation='linear')  # Capa de salida (Q-valores)
    ])

    # Función para elegir la acción según el estado actual
    def act(state):
        state = np.array(state, dtype=np.float32)  # Convertir a array
        state = np.reshape(state, [1, state_size])  # Ajustar forma
        q_values = model(state)  # Predecir Q-valores
        return np.argmax(q_values[0])  # Elegir acción con mayor Q-valor

    # Parámetros para el aprendizaje por refuerzo
    gamma = 0.99  # Factor de descuento (importancia de recompensas futuras)
    epsilon = 1.0  # Probabilidad de explorar (al principio es alta)
    epsilon_min = 0.1  # Límite inferior de exploración
    epsilon_decay = 0.995  # Factor de reducción de epsilon
    learning_rate = 0.001  # Tasa de aprendizaje
    optimizer = tf.keras.optimizers.Adam(learning_rate)  # Optimizador

    # Función para entrenar la red neuronal con la experiencia obtenida
    def train_model(state, action, reward, next_state, done):
        target = reward  # Valor objetivo inicial
        if not done:
            next_state = np.array(next_state, dtype=np.float32)
            print(f"train {next_state}")
            next_state = np.reshape(next_state, [1, state_size])
            target += gamma * np.max(model(next_state)[0])  # Suma recompensa futura

        # Calcular el gradiente y actualizar la red
        with tf.GradientTape() as tape:
            state = np.array(state, dtype=np.float32)
            state = np.reshape(state, [1, state_size])
            q_values = model(state)
            loss = tf.reduce_mean(tf.square(target - q_values[0][action]))  # Error cuadrático medio

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # --- Bucle principal de entrenamiento ---
    episode = 0  # Contador de episodios

    # Configurar gráfico interactivo para visualizar el progreso
    plt.ion()  # Activar modo interactivo
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Puntuación Total")
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Puntuación Total')
    ax.set_title('Progreso del Agente - Puntuación por Episodio')

    # Lista para guardar la recompensa total de cada episodio
    episode_rewards = []

    # Entrenamiento mientras la ventana del entorno esté abierta
    while env.window_open:
        state = env.reset()  # Reiniciar entorno y obtener estado inicial
        state = np.array(state, dtype=np.float32)
        state = np.reshape(state, [1, state_size])
        done = False
        truncated = False
        total_reward = 0

        # Ejecutar pasos dentro del episodio
        while  env.window_open:
            # Elegir acción: exploración o explotación
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()  # Acción aleatoria (exploración)
            else:
                action = act(state)  # Acción según la red (explotación)

            # Realizar la acción y obtener el siguiente estado y recompensa
            next_state, reward, done, truncated, info = env.step(action)

            # Entrenar la red con la experiencia obtenida
            train_model(state, action, reward, next_state, done)

            state = next_state  # Actualizar el estado actual
            print(f"Valor variable reward = {reward}")
            total_reward = reward  # Acumular recompensa (puedes cambiar a total_reward += reward si quieres la suma)

            # Actualizar el gráfico con la recompensa
            if len(episode_rewards) >= 0:
                line.set_xdata(np.arange(len(episode_rewards)))
                line.set_ydata(episode_rewards)
                ax.relim()
                ax.autoscale_view(True, True, True)
                plt.pause(0.01)  # Pausa breve para actualizar el gráfico

            env.render(mode='human')  # Mostrar el entorno visualmente

            # Si termina el episodio (por done o truncated)
            if done or truncated:
                episode_rewards.append(total_reward)  # Guardar recompensa
                print(f"total_reward: {total_reward}. Episodio done: {episode} Recompensa total = {episode_rewards}")
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay  # Reducir exploración
                episode += 1
                break

    # Si la ventana se cierra, mostrar mensaje final
    if not env.window_open:
        print(f"Episodio not window_open: {episode} Recompensa total = {episode_rewards}")

    # Cerrar el gráfico y mostrarlo al final
    plt.ioff()
    plt.close()
    plt.show()

    env.close()  # Cerrar el entorno después de todos los episodios

    # --- Ejemplo de método aleatorio (no aprende, solo explora) ---
    """i_episode = 0
        # Número de episodios que quieres ejecutar
        while env.window_open:
            state = env.reset()  # Reiniciar el entorno
            done = False

            while not done and env.window_open:
                # Tomar una acción aleatoria
                action = env.action_space.sample()

                # Realizar el paso en el entorno
                state, reward, done, truncated, info = env.step(action)
                #state, reward, done, info = env.step(action)

                # Llamada a render para mostrar el entorno visualmente
                env.render(mode='human')

                if done:
                    i_episode += 1
                    print(f"Episodio: {i_episode}")
                    break
                
        # Cerrar el entorno al final
        env.close()"""