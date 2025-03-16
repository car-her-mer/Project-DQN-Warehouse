from Warehouse import MiEntorno

# Registrar el entorno personalizado
import gym
import numpy as np
import tensorflow as tf
from keras import layers
from gym.envs.registration import register

# Registrar el entorno personalizado
register(
    id="MiEntorno-v1",  # Nombre único
    entry_point="Warehouse:MiEntorno",
)

# Crear una instancia del entorno
env = gym.make("MiEntorno-v1")

# Metodo DQN (al ser un espacio de observación continuo, es mejor utilizar DQN en vez de Q-learning)
"""
- Red Neuronal: La red neuronal en model predice los valores Q para cada acción a partir del estado actual.
- Exploración y Explotación: Al principio, el agente explora el entorno eligiendo acciones aleatorias (epsilon = 1.0), y con el tiempo, 
    reduce la exploración (epsilon decrece) y se enfoca más en explotar lo aprendido.
- Entrenamiento: La red neuronal se entrena con el método train_model, usando el error cuadrático medio entre el Q-valor predicho y el Q-valor objetivo calculado.
- Bucle de Episodios: El agente juega varios episodios en el entorno y va mejorando su política."""
# Parámetros del entorno
state_size = env.observation_space.shape[0]  # Tamaño del estado (dimensión del vector)
action_size = env.action_space.n  # Número de acciones posibles (discreto, como en el caso de una red neuronal con política discreta)

# Modelo DQN (red neuronal)
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(state_size,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(action_size, activation='linear')  # Acción con salida lineal (Q-valor)
])

# Funciones para el DQN
def act(state):
    # Usa la red neuronal para predecir las Q-values
    state = np.array(state, dtype=np.float32)  # Convertir tupla a array de numpy
    state = np.reshape(state, [1, state_size])
    q_values = model(state)
    return np.argmax(q_values[0])  # Escoge la acción con el Q-valor más alto

# Parámetros del aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Probabilidad de explorar al principio
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Función para entrenar la red neuronal
def train_model(state, action, reward, next_state, done):
    target = reward
    if not done:
        next_state = np.array(next_state, dtype=np.float32)  # Convertir tupla a array de numpy
        next_state = np.reshape(next_state, [1, state_size])
        target += gamma * np.max(model(next_state)[0])

    with tf.GradientTape() as tape:
        state = np.array(state, dtype=np.float32)  # Convertir tupla a array de numpy
        state = np.reshape(state, [1, state_size])  # Asegúrate de que state tenga la forma correcta
        q_values = model(state)
        loss = tf.reduce_mean(tf.square(target - q_values[0][action]))  # Error cuadrático medio

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Bucle principal de entrenamiento
episode = 0
#for episode in range(1000):  # Número de episodios
while env.window_open:
    state = env.reset()
    state = np.array(state, dtype=np.float32)  # Convertir tupla a array de numpy
  
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done and env.window_open:
        # Elige una acción
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = act(state)  # Explotación

        # Ejecuta la acción
        print(env.step(action))
        next_state, reward, done, truncated, info = env.step(action)
        #print(f"Contenido de state: {next_state}")
        #print(f"Tipo de state antes de convertir a tensor: {type(next_state)}")
        #print(f"Forma de state antes de convertir a tensor: {next_state.shape}")
        next_state = np.reshape(next_state, [1, state_size])

        # Entrena el modelo
        train_model(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Llamada a render para mostrar el entorno visualmente
        env.render(mode='human')

        if done:
            #print(f"Episode {episode+1}: Total reward: {total_reward}")
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay  # Decae epsilon para reducir la exploración


# Cerrar el entorno
env.close()

# Metodo basado en exploración aleatoria, no aprende ni se optimiza
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