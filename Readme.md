# **Proyecto Deep Reinforcement Learning - Optimización de rutas.**

## **Indice**
1. [Problema.](#1-problema)
2. [Idea.](#2-idea)
3. [Tecnologias.](#3-tecnologias)
4. [Proyecto.](#4-proyecto)

## 1. **Problema.**
<small>[Volver ↥](#indice)</small>

El problema que se busca solucionar sería el de **optimización de rutas**, donde el agente (robot, reparto, uber, etcetera) encuentra el camino más eficiente del punto A al punto B.
Actualmente se resuelve con tres enfoques tradicionales: *Grafos, optimización y heuristicas.*

### **Optimización (Ejemplo: Dijkstra)**
- **Concepto**: La optimización en el contexto de rutas significa encontrar el camino más eficiente entre dos puntos en una red. Piensa en un robot que necesita llegar del punto A al punto B en un almacén: la optimización busca la forma más rápida o menos costosa de hacer ese recorrido.
- **Ejemplo Conceptual**: Imagina que el robot tiene un mapa y usa un “algoritmo de optimización” como Dijkstra para saber qué camino tomar, eligiendo siempre la opción más corta en cada paso hasta llegar al destino.
- **Limitación**: Este tipo de algoritmo supone que el entorno es estático (es decir, que no cambian las rutas o las distancias), y no tiene capacidad para adaptarse a cambios imprevistos como obstáculos.

### **Heurísticas (Ejemplo: A)**
- **Concepto**: Una heurística es una “regla rápida” que ayuda a reducir el tiempo de búsqueda de una solución sin garantizar que sea perfecta, sino lo suficientemente buena. Este enfoque se usa cuando no es práctico o posible calcular la ruta exacta, especialmente en redes grandes o complejas.
- **Ejemplo Conceptual**: El robot, en lugar de revisar todas las rutas posibles, usa una heurística, como “moverse hacia el objetivo si parece que está más cerca”, para reducir la cantidad de rutas que considera. El algoritmo A* usa una combinación de la distancia real más una “estimación” de cuánto falta para llegar al destino, lo que le permite encontrar caminos buenos más rápido que con optimización pura.
- **Limitación**: Aunque es más rápido que Dijkstra en algunos casos, A* aún requiere que el entorno sea relativamente estable y no se adapta a cambios en tiempo real.

### **Grafos**
- **Concepto**: Un grafo es una estructura matemática que representa elementos (llamados nodos) y sus conexiones (llamadas aristas). Los métodos basados en grafos ayudan a representar mapas de rutas o conexiones de manera que se puedan aplicar algoritmos para encontrar rutas.
- **Ejemplo Conceptual**: Imagina el almacén como una serie de puntos de interés (estanterías, áreas de recogida, etc.), conectados por caminos. Cada punto es un nodo, y cada conexión es una arista. Los algoritmos de grafos, como Dijkstra o A*, pueden entonces usarse para buscar el camino óptimo de un punto a otro en este “mapa de nodos y conexiones”.
- **Limitación**: Los grafos representan bien conexiones estáticas, pero no son dinámicos, por lo que si se bloquea una conexión, el algoritmo necesita volver a calcular todo desde cero.

## 2. **Idea.**
<small>[Volver ↥](#indice)</small>
### **DQN (Deep Q-Learning)**
- **Concepto General**: DQN es una técnica de Reinforcement Learning (RL) que permite a un agente (como un robot) aprender qué acciones debe tomar en cada situación para maximizar sus recompensas a lo largo del tiempo. A diferencia de los métodos tradicionales, DQN permite al agente aprender mediante prueba y error, explorando y mejorando con cada experiencia.

- **Cómo Funciona Conceptualmente**: Imagina que el robot en el almacén tiene una tabla que le dice qué tan buena es cada acción en cada posición o situación. Esta tabla se llama tabla Q y le dice al robot qué tan "recompensante" será cada acción en función de sus experiencias pasadas.

- **Ejemplo**: si en el pasado el robot tomó una acción que le acercó a su objetivo, se guarda esa acción como "buena" en su memoria, y si tomó una acción que lo alejó, aprende a evitarla.
A medida que el robot explora más, esta tabla Q se vuelve más completa y precisa, ayudando al robot a tomar decisiones más inteligentes en el futuro.

- **¿Qué aporta el Deep Learning?**: En entornos complejos, como un almacén con muchas rutas, la tabla Q se vuelve enorme y difícil de gestionar. Aquí es donde entra el Deep Learning: en lugar de guardar cada acción posible, DQN utiliza una red neuronal para aprender patrones y estimar la tabla Q, almacenando solo la información esencial. Esto permite que el robot maneje entornos grandes y complejos, sin sobrecargar su memoria.

- **Ventajas de DQN**
  - **Adaptabilidad**: A diferencia de los métodos tradicionales de optimización o heurísticas, el DQN permite que el robot se adapte a los cambios en el entorno (como obstáculos nuevos o rutas bloqueadas) sin necesidad de volver a calcular todo desde cero.
  - **Aprendizaje Continuo**: DQN permite que el robot aprenda de cada acción que toma, incluso si esa acción fue una “mala decisión”. Esto le ayuda a mejorar con cada experiencia, algo que no sucede en los enfoques tradicionales.

- **Limitaciones.**
  - **Entrenamiento Intensivo**: El DQN requiere mucho entrenamiento antes de que el robot tome decisiones óptimas. En entornos complejos, el entrenamiento puede tomar tiempo y ser intensivo computacionalmente.
  - **Exploración y Explotación**: DQN necesita explorar varias opciones antes de poder explotar sus conocimientos para tomar siempre las mejores decisiones. Esta fase de exploración puede llevar a que el robot tome decisiones subóptimas al inicio.

## 3. **Tecnologias.**
<small>[Volver ↥](#indice)</small>
- **Python.**
- **Pygame:** Crear mi propio entorno en OpenAI GYM.
- **OpenAI GYM:** Entorno donde entrenar al modelo.
- **TensorFlow:** para la parte de DL (Deep Learning)
- **Deep Q Network:** para el modelo de DRL (Deep Reinforcement Learning)

## 4. **Proyecto.**
<small>[Volver ↥](#indice)</small>
(En proceso)


