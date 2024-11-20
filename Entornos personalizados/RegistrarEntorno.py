from Warehouse import MiEntorno

# Registrar el entorno personalizado
import gym
from gym.envs.registration import register

# Registrar el entorno personalizado
register(
    id="MiEntorno-v0",  # Nombre único
    entry_point="Warehouse:MiEntorno",
)

# Crear una instancia del entorno
env = gym.make("MiEntorno-v0")

i_episode = 0

# Número de episodios que quieres ejecutar
while env.window_open:
    state = env.reset()  # Reiniciar el entorno
    done = False

    while not done and env.window_open:
        # Tomar una acción aleatoria
        action = env.action_space.sample()

        # Realizar el paso en el entorno
        state, reward, done, truncated, info = env.step(action)

        # Llamada a render para mostrar el entorno visualmente
        env.render(mode='human')

        if done:
            i_episode += 1
            print(f"Episodio: {i_episode}")
            break
        
# Cerrar el entorno al final
env.close()