# Archivo principal del proyecto.
# Este archivo es el punto de entrada de la aplicación.
# Aquí se importa y lanza el menú principal, y según la opción elegida por el usuario,
# se inicia el entorno de entrenamiento de OpenAI Gym.
# Toda la navegación y lógica general del flujo de la app comienza aquí.

from Menus import menu_principal
from Entorno.Entrenamiento import IniciarEntorno

# Mostrar el menú principal y comprobar la opción elegida
if menu_principal.mostrar_menu() == "entrenar":
    # Si el usuario elige entrenar, se inicia el entorno de entrenamiento
    mi_entorno = IniciarEntorno()  # Llama a la función que ejecuta el entorno y el agente
