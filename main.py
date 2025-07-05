# Este es el archivo que ejecutará tu aplicación. 
# Aquí se importarán las funciones y clases de otros archivos,
# y se gestionará la lógica general de tu proyecto. 
# Al ser el archivo principal, este manejará la ejecución del menú, 
# los eventos del usuario y el paso de un menú a otro. 
# Además, iniciará la ejecución del entorno de OpenAI Gym cuando el usuario lo decida.

# se cargan los menús de la interfaz gráfica, 
# llamando a las funciones de menu_principal.py y menu_info.py.

from Menus import menu_principal
from Entorno.Entrenamiento import IniciarEntorno

if menu_principal.mostrar_menu() == "entrenar":
    # Crear una instancia del entorno
    mi_entorno = IniciarEntorno()  # Instancia de la clase que contiene el código del entorno
