import numpy as np

# Datos

X = np.array([
     [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
     [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# La humedadad va aprox de 0 a 100 y la temperatura de 0 a 50
# Dividimos cada columna por su valor maximo esperado

escala = np.array([100, 50])
X_normalizado = X / escala

print ("Datos normalizados:")
print (X_normalizado)

# Neurona

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def entrenar(tasa_aprendizaje, epocas, mostrar=False):
    # Se utiliza la misma semilla, asi todos los experimentos comienzan igual
    # y solo se cambia el parametro que probamos

    rng = np.random.default_rng(42)
    pesos = rng.normal(0, 0.5, size=(2, 1)) # Peso humedad, peso temperatura
    sesgo = 0.0

    for epoca in range(epocas + 1):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)
        error = probabilidad - y
        ecm = np.mean(error ** 2)

        if mostrar and epoca % (epocas // 10) == 0:
            print(f"Época {epoca:>6} | Error cuadrático medio: {ecm:.6f}")

        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos = pesos - tasa_aprendizaje * gradiente_pesos
        sesgo = sesgo - tasa_aprendizaje * gradiente_sesgo

    # Error final con los pesos actualizados
    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    ecm_final = np.mean((probabilidad - y) ** 2)
    return pesos, sesgo, ecm_final

def predecir(humedad, temperatura, pesos, sesgo):
    # Los datos nuevos se normalizan con la Misma escala
    entrada = np.array([humedad, temperatura], dtype=float) / escala
    return sigmoide(entrada @ pesos + sesgo).item()

# Entrenamiento Base
print("\n=== ENTRENAMIENTO BASE (0.5, 10000 épocas) ===")
pesos, sesgo, ecm = entrenar(0.5, 10000, mostrar=True)

print(f"\nPeso humedad:      {pesos[0, 0]:.4f}")
print(f"Peso temperatura:  {pesos[1, 0]:4f}")
print(f"Sesgo:             {sesgo:.4f}")
print(f"Error final:       {ecm:.6f}")

prob_train = sigmoide(X_normalizado @ pesos + sesgo)
print("\nProbabilidad por caso:")
for i in range (len(X)):
    print(f"Caso {i+1:>2} | Hum {X[i,0]:>3.0f}% | Temp {X[i,1]:>2.0f}°C "
          f"| Prob {prob_train[i,0]:.4f} | Esperado {int(y[i,0])}")

# Pruebas con condiciones nuevas
print("\n=== CONDICIONES NUEVAS ===")
nuevos = [(75, 30), (45, 34), (25, 22), (50,25), (30, 40)]
for h, t in nuevos:
    p = predecir(h, t, pesos, sesgo)
    decision = int(p >= 0.5)
    print(f"Hum {h}% | Temp {t}°C | Prob {p:.4f} | Decisión {decision}")

# Experimenteos
print("\n=== EXPERIMENTOS ===")
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),                                  
]

print(f"{'Experimento':<20} {'Épocas':>7} {'Tasa':>6} {'Error final':>12} "
      f"{'Aciertos':>9} {'Prob(45,34)':>12}")

for nombre, ep, tasa in experimentos:
    w, b, e = entrenar(tasa, ep)
    pred = (sigmoide(X_normalizado @ w + b) >= 0.5).astype(int)
    aciertos = int(np.sum(pred == y))
    p = predecir(45, 34, w, b)
    print(f"{nombre:<20} {ep:>7} {tasa:>6} {e:>12.6f} {aciertos:>7}/10 {p:>12.4f}")

# Prueba con umbrales (sin entrenar, usando el modelo base)
print("\n=== UMBRALES 0.4 / 0.5 / 0.6 ===")
print("Datos de entrenamiento:")
for i in range(len(X)):
    p = prob_train[i, 0]
    print(f"Caso {i+1:>2} | Prob {p:.4f} | "
          f"0.4 -> {int(p >= 0.4)} | 0.5 -> {int(p >= 0.5)} | 0.6 -> {int(p >= 0.6)}")

print("\n=== Condiciones nuevas: ===")
for h, t in nuevos:
    p = predecir(h, t, pesos, sesgo)
    print(f"Hum {h}% Temp {t}°C | Prob {p:.4f} | "
          f"0.4 -> {int(p >= 0.4)} | 0.5 -> {int(p >= 0.5)} | 0.6 -> {int(p >= 0.6)}")
