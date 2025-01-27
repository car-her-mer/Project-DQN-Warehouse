from Warehouse_manual import MiEntorno

# Registrar el entorno personalizado
import gym
from gym.envs.registration import register

env = MiEntorno()
state, _ = env.reset()
done = False

while env.window_open:
    env.render()  # Esto ahora también maneja la interacción del usuario
env.close()
