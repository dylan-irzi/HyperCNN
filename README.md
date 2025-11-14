# HyperCNN  
**Hyperbolic-topology convolutions for ultra-short paths**  
**Convoluciones con topología hiperbólica para caminos ultra-cortos**

---

## 🇺🇸 English Version

Deep CNNs have achieved outstanding results in image classification, but often with heavy compute and memory costs. Residual networks improved gradient flow, but they don’t ensure globally efficient connectivity. Transformers, while powerful, scale quadratically and aren’t suited for edge deployment.

**HyperCNN** introduces a mathematically designed connectivity pattern inspired by the small-world phenomenon. Any block can reach any other in just **2–4 hops**, without attention, tokens, or graph ops — fully compatible with classic, efficient CNN implementations.

---

## 🚀 Highlights

### 🔭 Hyperbolic Topology  
A deterministic base-4 indexing system guarantees a tiny network diameter:  
**D ≤ log₄(N)**

### ⚡ Efficient Implementation  
Zero inference overhead — all connections are precomputed and mapped to standard convolutions + tensor additions.

### 🏆 Performance  
Outperforms **MobileNetV2** and **ShuffleNetV2** on CIFAR-10 with comparable or fewer parameters.

### 🔧 Compatibility  
Fully supports **PyTorch**, **mixed precision**, and **ONNX export**.

---

## 🧠 2. Architecture

### 2.1 Short-Path Connectivity  
Each of the **N blocks** receives a 4-digit base-4 identifier (example: `[0,1,3,2]`).  
Two blocks connect if they share **at least one digit in the same position**.

**Examples**

- `[0,0,0]` vs `[3,3,3]` → no shared digits → ❌ no connection  
- `[1,1,1]` vs `[2,0,0]` → no shared digits → ❌ no connection  
- `[1,1,1]` vs `[2,1,0]` → share “1” → ✔️ connected

**Benefits**

- High local clustering  
- Short global paths: **≤ ⌈log₄(N)⌉**  
- Fully deterministic  

### 🔑 Network Diameter

| N Blocks | Diameter D |
|---------|------------|
| 16      | 2          |
| 32      | 3          |
| 64      | 3          |

---

# 🇪🇸 Versión en Español

Las CNN profundas han logrado resultados fuertes en clasificación de imágenes, pero suelen requerir bastante cómputo y memoria. ResNet ayudó con el flujo de gradiente, pero no garantiza eficiencia global. Transformers, aunque potentes, escalan cuadráticamente y no son ideales para dispositivos edge.

**HyperCNN** propone una conectividad diseñada matemáticamente, inspirada en el fenómeno *small-world*: cualquier bloque puede alcanzar a cualquier otro en **2–4 saltos**, sin atención, sin tokens y sin operaciones especiales. Todo funciona con convoluciones clásicas y eficientes.

---

## 🚀 Highlights

### 🔭 Topología Hiperbólica  
Un sistema determinista basado en índices en base 4 asegura un diámetro mínimo:  
**D ≤ log₄(N)**

### ⚡ Implementación Eficiente  
No añade costo en inferencia: las conexiones se precomputan y usan convoluciones estándar + sumas de tensores.

### 🏆 Rendimiento  
Supera a **MobileNetV2** y **ShuffleNetV2** en CIFAR-10 con un número de parámetros comparable o menor.

### 🔧 Compatibilidad  
Compatible con **PyTorch**, **mixed precision** y **exportación ONNX**.

---

## 🧠 2. Arquitectura

### 2.1 Conectividad de Camino Corto  
A cada uno de los **N bloques** se le asigna un identificador de 4 dígitos en base 4 (ejemplo: `[0,1,3,2]`).  
Dos bloques se conectan si comparten **al menos un dígito en la misma posición**.

**Ejemplos**

- `[0,0,0]` vs `[3,3,3]` → sin coincidencias → ❌ sin conexión  
- `[1,1,1]` vs `[2,0,0]` → sin coincidencias → ❌ sin conexión  
- `[1,1,1]` vs `[2,1,0]` → comparten “1” → ✔️ conexión

**Ventajas**

- Alta cohesión local  
- Caminos globales cortos: **≤ ⌈log₄(N)⌉**  
- Arquitectura completamente determinista  

### 🔑 Diámetro de Red

| N Bloques | Diámetro D |
|-----------|------------|
| 16        | 2          |
| 32        | 3          |
| 64        | 3          |

N = 32 → D = 3

N = 64 → D = 3
