# Neurona artificial para el riego de plantas

Taller de IA Generativa - Universidad Santo Tomás, Tunja.

## Descripción del problema

Se construyó una neurona artificial con NumPy que decide si una planta necesita riego a partir de dos entradas:
la humedad del suelo (0 a 100 %) y la temperatura ambiental (aprox. 0 a 50 °C). La salida es 1 (regar) o 0
(no regar). La neurona tiene dos pesos, un sesgo y función de activación sigmoide, y se entrena con descenso
de gradiente sobre diez ejemplos didácticos.

## Cómo ejecutarlo

```
uv sync
uv run main.py
```

El programa muestra los datos normalizados, el entrenamiento base, las probabilidades por caso, las cinco predicciones
nuevas, los ocho experimentos y la prueba de umbrales. Para que los experimentos sean comparables, todos parten de los
mismos pesos iniciales (semilla 42).

## Resultado del entrenamiento base (tasa 0.5, 10000 épocas)

| Parámetro | Valor |
|---|---|
| Peso de la humedad | -33.2678 |
| Peso de la temperatura | 0.2708 |
| Sesgo | 14.7906 |
| Error cuadrático medio final | 0.005369 |

El peso de la humedad es negativo: a más humedad, menor probabilidad de regar. El de la temperatura es positivo pero muy
pequeño: más calor empuja un poco hacia regar, aunque la neurona casi no lo tiene en cuenta.

Probabilidades en los datos de entrenamiento:

| Caso | Humedad | Temperatura | Probabilidad | Esperado |
|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0000 | 0 |
| 2 | 70 % | 22 °C | 0.0002 | 0 |
| 3 | 65 % | 28 °C | 0.0013 | 0 |
| 4 | 55 % | 25 °C | 0.0332 | 0 |
| 5 | 50 % | 32 °C | 0.1584 | 0 |
| 6 | 40 % | 30 °C | 0.8384 | 1 |
| 7 | 35 % | 25 °C | 0.9638 | 1 |
| 8 | 30 % | 32 °C | 0.9932 | 1 |
| 9 | 20 % | 35 °C | 0.9998 | 1 |
| 10 | 10 % | 38 °C | 1.0000 | 1 |

## Pruebas con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0000 | 0 (no regar) |
| 45 % | 34 °C | 0.5011 | 1 (regar) |
| 25 % | 22 °C | 0.9986 | 1 (regar) |
| 50 % | 25 °C | 0.1534 | 0 (no regar) |
| 30 % | 40 °C | 0.9935 | 1 (regar) |

## Experimentos con los parámetros

| Experimento | Épocas | Tasa | Error final | Aciertos | Prob. (45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.005369 | 10/10 | 0.5011 | Adecuado |
| Pocas épocas | 100 | 0.5 | 0.089027 | 10/10 | 0.5527 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.030278 | 10/10 | 0.5645 | Aceptable, aún mejorable |
| Más épocas | 20000 | 0.5 | 0.002798 | 10/10 | 0.4867 | Bueno, mejora pequeña |
| Tasa pequeña | 10000 | 0.01 | 0.066628 | 10/10 | 0.5686 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.019444 | 10/10 | 0.5441 | Algo lento |
| Tasa alta | 10000 | 1.0 | 0.002797 | 10/10 | 0.4867 | Rápido y estable |
| Tasa muy alta | 10000 | 2.0 | 0.001396 | 10/10 | 0.4722 | Muy rápido, sin inestabilidad |

## Prueba adicional con el umbral

Sin reentrenar, con los pesos de la prueba base:

- En los diez datos de entrenamiento no cambió ninguna respuesta con umbrales de 0.4, 0.5 y 0.6, porque todas las probabilidades
están lejos de esos valores (la más cercana es 0.1584 y la más baja de las positivas es 0.8384).
- En las condiciones nuevas solo cambió el caso de 45 % de humedad y 34 °C (probabilidad 0.5011): con 0.4 y 0.5 la decisión es
regar, con 0.6 es no regar.

Cambiar el umbral no modifica los pesos porque el umbral se aplica después del entrenamiento, sobre la probabilidad ya calculada.
 Solo cambia dónde cortamos para pasar de probabilidad a 0 o 1; los pesos y el sesgo siguen siendo los mismos.

## Análisis

**1. ¿Por qué fue necesario normalizar la humedad y la temperatura?**
Porque las dos variables están en escalas distintas la humedad llega a 100 y la temperatura a unos 50. Si se utilizan sin normalizar,
la variable de números más grandes domina la suma ponderada, el valor de z se vuelve enorme y la sigmoide se satura, con lo que el
gradiente casi desaparece y el aprendizaje se vuelve lento o torpe.

**2. ¿En qué operaciones se utilizó X_normalizado y para qué se conservó X?**
X_normalizado se utilizó en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @
gradiente_z`). X se conservó para mostrar los valores originales (80 %, 18 °C, etc.) en las tablas, porque así los resultados se
pueden interpretar facilmente.

**3. ¿Qué ocurrió al utilizar solamente 100 épocas?**
La neurona acertó los 10 casos, pero con muy poca seguridad, el error quedó en 0.089, bastante más alto que 0.0054 de la prueba base.
Las probabilidades todavía estaban muy cerca de la zona indecisa. Clasificaba bien, pero el entrenamiento no había terminado,fue
insuficiente.

**4. ¿Más épocas siempre produjeron una mejora importante?**
No, de 100 a 1000 épocas el error bajó mucho (0.089 a 0.030) y de 1000 a 10000 también (0.030 a 0.0054). Pero al pasar de 10000 a
20000 solo bajó de 0.0054 a 0.0028, y los aciertos seguían siendo 10 de 10.

**5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?**
Con 0.01 la neurona aprendió muy lento, después de 10000 épocas el error era 0.0666, parecido al de entrenar con 0.5 durante
apenas 100 épocas. Los pasos eran tan pequeños que le faltaba recorrido.

**6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?**
No hubo inestabilidad. Con 1.0 y con 2.0 el error bajó rapidamente y terminó incluso menor (0.0028 y 0.0014).Esperaba que se
descontrolara con 2.0, pero los datos son pocos y están bien separados, así que la neurona aguantó pasos grandes. Aun así,
eso no significa que una tasa alta sirva siempre, con otros datos podría oscilar o no converger.

**7. ¿Qué representa el signo del peso correspondiente a la humedad?**
Es negativo (-33.27), lo que quiere decir que a mayor humedad del suelo, la probabilidad de regar baja. Tiene sentido: un suelo
mojado no necesita agua.

**8. ¿Qué representa el signo del peso correspondiente a la temperatura?**
Es positivo (0.27): a más calor, la probabilidad de regar sube un poco. Pero el valor es muy pequeño, así que la neurona casi no
usa la temperatura.

**9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?**
Porque al final hay que tomar una acción: regar o no regar. La sigmoide entrega un número entre 0 y 1 que expresa qué tan probable
es el riego, pero eso no es una decisión. El umbral es la regla que convierte la probabilidad en una respuesta concreta, y se puede
ajustar según el costo de equivocarse.

**10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?**
Tiene solo dos variables, cuando en la realidad influyen el tipo de planta, el tipo de suelo, la lluvia, el viento, la hora del día,
etc. Se entrenó con apenas diez datos inventados y se evaluó con esos mismos datos, así que no sabemos qué tan bien generaliza. Además,
una sola neurona solo traza una frontera recta entre regar y no regar, y su respuesta es de sí o no, sin decir cuánta agua necesita la
planta.
