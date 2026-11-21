# Parte I – Los 4 filtros: explicación matemática y algorítmica

Proyecto Integrador de Visión Artificial (UPS). Este documento explica **cada decisión** de la PARTE A de `ProyectoVision.cpp` para poder defenderlo en la sustentación. El quinto filtro (el artístico propio) queda pendiente y se construirá combinando estos cuatro.

> **Declaración de uso de IA (requisito de la guía):** el código de `ProyectoVision.cpp` fue generado con asistencia de Claude (Anthropic) y adaptado a este proyecto. Este documento explica su lógica. Las funciones de OpenCV se citan en la sección 7; la estructura de la ventana (lienzo con `copyTo` sobre `Rect`, trackbars, bucle `while(3==3)`) viene de los ejemplos 6–9 de la Unidad 2 del curso.

---

## 0. Resumen de los 4 filtros

| # | Filtro | Tipo | Qué hace | Ejemplo de clase donde se vio |
|---|--------|------|----------|-------------------------------|
| 1 | Gaussiano | Convolución lineal (pasa-bajos) | Suaviza, reduce ruido | Ej. 6, 7, 8, 9 |
| 2 | Sobel | Convolución lineal (derivada) | Detecta bordes (gradiente) | Ej. 6, 7, 8 |
| 3 | Ecualización / CLAHE | Operación de punto sobre histograma | Aumenta el contraste | Ej. 4, 5 (y Lab en Ej. 3) |
| 4 | Morfología | No lineal (mín/máx) | Erosión, dilatación, apertura, cierre, gradiente… | Ej. 7 |

Los filtros 1 y 2 son **lineales** (se expresan como convolución). El 3 es una **transformación de intensidades** basada en estadística (histograma = distribución de frecuencias). El 4 es **no lineal**. Esa variedad es útil para el efecto artístico.

---

## 1. Base común: correlación/convolución discreta

Para un kernel `K` de tamaño `k×k` (con `k = 2r+1`), `cv::filter2D` calcula para cada píxel:

```
dst(x, y) = Σ_{i=-r..r} Σ_{j=-r..r}  K(i, j) · src(x+i, y+j)
```

- Es matemáticamente una **correlación**; la convolución verdadera voltea el kernel. Para el Gaussiano (simétrico) es idéntico. Para Sobel solo cambia el signo del gradiente, y como luego usamos la magnitud `√(Gx²+Gy²)` el resultado no cambia.
- **Costo:** `k²` multiplicaciones por píxel por canal. Un kernel 31×31 son 961 multiplicaciones por píxel: así se logra "saturar" la CPU en el benchmark que pide la guía.
- **Bordes:** los píxeles fuera de la imagen se rellenan con `BORDER_REFLECT_101` (reflejo sin repetir el píxel del borde: `gfedcb|abcdefgh|gfedcba`), que evita el borde oscuro que produciría rellenar con ceros.
- Los kernels son siempre **impares** (`forzarImpar`): hace falta un píxel central para que el filtro no desplace la imagen.

---

## 2. Filtro Gaussiano (suavizado)

**Idea.** Cada píxel pasa a ser un promedio ponderado de su vecindario, donde los vecinos cercanos pesan más que los lejanos. Elimina ruido de alta frecuencia a costa de desenfocar.

**Matemática.** La gaussiana 2D continua es

```
G(x,y) = 1/(2πσ²) · exp( −(x²+y²) / (2σ²) )
```

En el código (`kernelGaussiano`) se evalúa `exp(−(x²+y²)/(2σ²))` para `x,y ∈ [−r, r]` y se **normaliza dividiendo por la suma**, así `Σ K = 1`. Con eso la constante `1/(2πσ²)` se cancela y se cumple que **el brillo medio de la imagen no cambia** (prueba automática: diferencia de media 0.001).

**¿Por qué esta función?**
- Es **isotrópica** (no favorece ninguna dirección) y su decaimiento suave no produce artefactos de "anillado" (a diferencia de un promedio uniforme).
- Es **separable**: `G(x,y) = g(x)·g(y)`; podría aplicarse como dos pasadas 1D (`sepFilter2D`, costo `2k` en vez de `k²`). Aquí se usa la forma 2D completa con `filter2D` (la que pide la rúbrica) y, de paso, es la más costosa → mejor para el benchmark. Es una decisión que conviene mencionar en la defensa.
- En frecuencia también es gaussiana (`exp(−σ²ω²/2)`): es un filtro **pasa-bajos** limpio. Mayor `σ` ⇒ corta frecuencias más bajas ⇒ más desenfoque.

**Parámetros.**
- `ksize`: tamaño del kernel (trackbar "Gauss kernel").
- `σ`: si es 0 se usa la regla de OpenCV `σ = 0.3·((k−1)/2 − 1) + 0.8` (k=3→0.8, k=5→1.1, k=9→1.4, k=31→5.0). Regla práctica: `k ≈ 6σ+1` para que el kernel cubra ±3σ (99.7 % del área) y los extremos valgan casi 0.

**Ejemplo numérico (k=3, σ=0.8).** Pesos sin normalizar: centro 1, lados `e^{-1/1.28}=0.4578`, esquinas `e^{-2/1.28}=0.2096`; suma 3.6698. Normalizado:

```
0.0571 0.1248 0.0571
0.1248 0.2725 0.1248
0.0571 0.1248 0.0571        (suma = 1)
```

**Validación.** Coincide con `cv::GaussianBlur` con diferencia máxima de 1 nivel de gris (esa función usa aritmética de punto fijo, de ahí el ±1).

---

## 3. Filtro Sobel (detección de bordes)

**Idea.** Un borde es un cambio brusco de intensidad, es decir, una zona donde la **derivada** (gradiente) de la imagen es grande.

**Matemática.** El gradiente es `∇I = (∂I/∂x, ∂I/∂y)`. En una imagen discreta la derivada se aproxima con diferencias finitas; la diferencia central es `(I(x+1) − I(x−1))/2`, pero amplifica el ruido. Sobel la combina con un **suavizado** binomial `[1 2 1]/4` en la dirección perpendicular:

```
Gx = (1/4)[1 2 1]ᵀ ⊗ (1/2)[−1 0 1]  =  1/8 · | −1  0  1 |        Gy = 1/8 · | −1 −2 −1 |
                                              | −2  0  2 |                    |  0  0  0 |
                                              | −1  0  1 |                    |  1  2  1 |
```

Es el **producto exterior** (columna × fila) de un vector de suavizado y un vector de derivada; por eso `kernelsSobel` usa `getDerivKernels` (entrega esos dos vectores 1D) y los multiplica `ky * kx.t()`. Para `ksize` mayor (5, 7, …) el vector de suavizado es la fila de Pascal (binomial ≈ gaussiana) y el de derivada es su derivada, de modo que **Sobel grande ≈ derivada de una gaussiana**: más inmune al ruido, bordes más gruesos.

Luego:

```
Magnitud   |∇I| = √(Gx² + Gy²)          Dirección   θ = atan2(Gy, Gx)
```

**Decisiones de implementación.**
- Se trabaja en **escala de grises** (la intensidad), porque un borde de color sin cambio de luminancia casi no se percibe.
- `filter2D(..., CV_32F, ...)`: con 8 bits los valores negativos se saturarían a 0 y se perdería la mitad de los bordes (los de transición claro→oscuro).
- Kernel **normalizado** (`/8` para 3×3): `Gx` queda en "niveles de gris por píxel", independiente de `ksize`. Un escalón de 0 a 255 da `Gx = 127.5`.
- `ganancia`: factor de visualización, la salida es `saturate_cast<uchar>(ganancia · |∇I|)`.
- **Diferencia con la clase (Ej. 6):** allí se usó `convertScaleAbs` en cada eje y `addWeighted(0.5, 0.5)`, que aproxima la magnitud con `(|Gx|+|Gy|)/2`. Aquí se usa la norma L2 exacta (`cv::magnitude`).

**Validación.** `Sobel 3×3` coincide exactamente con la matriz de arriba, y para 3, 5 y 7 coincide con `cv::Sobel` (diferencia < 0.001). Imagen constante ⇒ salida 0; escalón vertical ⇒ borde solo en el escalón.

---

## 4. Ecualización de histograma y CLAHE

**Idea.** Si la imagen usa solo una parte del rango de grises (poco contraste), se **redistribuyen** las intensidades para aprovechar todo el rango.

**Matemática (ecualización global).** Sea `h(k)` el número de píxeles con nivel `k` (0…255) y `N` el total.
```
p(k) = h(k)/N                         (función de masa de probabilidad)
c(k) = Σ_{j≤k} p(j)                   (función de distribución acumulada, CDF)
s(k) = round( 255 · (c(k) − c_min)/(1 − c_min) )     (nuevo nivel)
```
La CDF como transformación convierte una variable aleatoria en una aproximadamente **uniforme** (transformada integral de probabilidad), por eso el histograma resultante queda "plano". Es el vínculo con Estadística (variables aleatorias y distribuciones).

**Limitación:** es *global*: una zona muy oscura y otra muy clara se tratan con la misma curva, y el ruido de zonas planas se amplifica.

**CLAHE (Contrast Limited Adaptive Histogram Equalization).**
1. Divide la imagen en una rejilla `tiles × tiles` de bloques.
2. Calcula el histograma de cada bloque.
3. **Recorta** el histograma en `clipLimit · (píxeles del bloque / 256)` y reparte el exceso uniformemente entre los niveles; esto limita la pendiente de la CDF y por tanto cuánto se amplifica el ruido.
4. Ecualiza cada bloque con su CDF.
5. Interpola **bilinealmente** entre las curvas de los 4 bloques vecinos para que no se vean los bordes de bloque.

Valores típicos: `clipLimit` 2–4, `tiles` 8. (En el ejemplo 5 de clase se usó `clipLimit = 40`, que casi no recorta y se comporta como ecualización adaptativa pura, con mucho ruido.)

**¿Por qué en el canal L de CIELab?** Si se ecualizara B, G y R por separado se cambiarían las *proporciones* entre canales y los colores se distorsionarían. CIELab separa la **luminosidad L** de la **cromaticidad (a, b)**; se ecualiza solo `L` y se vuelve a BGR (en OpenCV 8 bits, `L ∈ [0,100]` se escala a `[0,255]`). El espacio Lab ya se vio en clase (Ej. 3, `COLOR_BGR2Lab`).

**Validación.** Sobre una imagen artificialmente comprimida a [100, 164] la desviación estándar sube de 2.9 a 65.8 (global) y de 2.9 a 7.9 (CLAHE con clip 4, que es deliberadamente más conservador).

---

## 5. Operaciones morfológicas

**Idea.** Operan sobre la **forma** de las estructuras usando un *elemento estructurante* `B` (una pequeña "sonda" de forma rectangular, cruz o elipse) que se desliza por la imagen. No son convoluciones: son operaciones **no lineales** de mínimo y máximo.

**Matemática (escala de grises).**
```
Erosión      (f ⊖ B)(x,y) = min_{(i,j)∈B} f(x+i, y+j)     → encoge lo claro, agranda lo oscuro
Dilatación   (f ⊕ B)(x,y) = max_{(i,j)∈B} f(x+i, y+j)     → agranda lo claro, encoge lo oscuro
Apertura     f ∘ B = (f ⊖ B) ⊕ B     → elimina objetos claros más pequeños que B (ruido "sal")
Cierre       f • B = (f ⊕ B) ⊖ B     → rellena huecos oscuros más pequeños que B (ruido "pimienta")
Gradiente    (f ⊕ B) − (f ⊖ B)       → contornos; el grosor depende del tamaño de B
Top-hat      f − (f ∘ B)             → detalles claros pequeños
Black-hat    (f • B) − f             → detalles oscuros pequeños
```
En una imagen binaria (0/255) esto equivale a las definiciones de teoría de conjuntos `A ⊖ B = {z | B_z ⊆ A}` y `A ⊕ B = {z | B̂_z ∩ A ≠ ∅}`.

**Propiedades que se verifican con `--pruebas`** (son buenas preguntas de defensa):
- Erosión **anti-extensiva** y dilatación **extensiva**: `erosión ≤ original ≤ dilatación`.
- `apertura ≤ original ≤ cierre`.
- La apertura es **idempotente**: abrir dos veces = abrir una vez.
- `gradiente = dilatación − erosión` (exacto).

**Elemento estructurante.** `getStructuringElement` con `MORPH_RECT` (cuadrado), `MORPH_CROSS` (cruz, el usado en el ejemplo 7) o `MORPH_ELLIPSE` (elipse, más isotrópica; es el valor por defecto). `iteraciones = n` equivale a aplicar la operación n veces (efecto de un elemento más grande). En imágenes de color se aplica por canal.

---

## 6. Diseño "pipeline dual" (Mat / UMat)

Cada filtro es una plantilla `template<class M>` con `M = cv::Mat` o `cv::UMat`:

- **`cv::Mat`** → ejecución tradicional en CPU.
- **`cv::UMat`** → *API Transparente (T-API)*: las funciones de OpenCV reciben `InputArray`/`OutputArray`; si el argumento es `UMat` y `cv::ocl::useOpenCL()` es verdadero, lanzan un *kernel* OpenCL en la GPU; si no, ejecutan la versión CPU. **El código del filtro es idéntico**, así la comparación del benchmark es justa.
- Las variables intermedias (`gris`, `gx`, `gy`, `lab`, …) son del mismo tipo `M` para que **todo el filtro se quede en la GPU** y no haya descargas implícitas a RAM.
- Medición correcta (se aplica en la demo y en las pruebas): (a) las órdenes OpenCL son **asíncronas**, hay que llamar `cv::ocl::finish()` antes de detener el reloj; (b) la primera llamada **compila** los kernels OpenCL, por eso se hacen ejecuciones de calentamiento; (c) la subida RAM→VRAM y la bajada VRAM→RAM se miden aparte porque cuentan para los FPS (la guía lo menciona explícitamente).
- La PARTE A de `ProyectoVision.cpp` (los filtros) no usa highgui, así que se puede copiar tal cual al código nativo de Android (NDK/JNI).

---

## 7. Qué se verificó y qué NO

Ejecutado en el entorno de desarrollo (OpenCV 4.6.0, C++17): **todas las pruebas automáticas pasan** (`./vision.bin --pruebas`):

- Kernels: suma = 1, simetría, máximo central (Gauss); Sobel == matriz conocida y == `cv::Sobel`.
- Gaussiano == `cv::GaussianBlur` (±1); propiedades de los 4 filtros.
- Equivalencia `Mat` vs `UMat`: diferencia máxima 0 en todos los filtros salvo CLAHE (1 nivel). Esta comprobación se hizo con OpenCL real usando el runtime **pocl** (OpenCL por software sobre la CPU), con `OPENCV_OPENCL_DEVICE=":CPU:"`.
- La demo interactiva se ejecutó en un display virtual con una imagen y con un vídeo que simula la cámara: los 4 filtros, los trackbars y el cambio CPU↔GPU con la tecla `g` funcionan. **No se probó con una webcam real** (el entorno de desarrollo no tiene cámara).

**No verificado:** el rendimiento en una GPU real. En el entorno de desarrollo no había GPU, y con pocl la "GPU" es la propia CPU (los tiempos salen peores y no significan nada). Los tiempos CPU vs GPU que van al informe **deben medirse en tu equipo** (`./vision.bin --pruebas`, sección D, o los milisegundos que muestra la demo); la guía pide una mejora de 3–5×. En Linux con GPU NVIDIA/AMD/Intel necesitas el driver con soporte OpenCL.

---

## 8. Cómo compilar y ejecutar (un solo archivo)

```bash
# Linux (Ubuntu/Debian):  sudo apt install build-essential cmake pkg-config libopencv-dev
cd proyecto_interciclo

# Opción 1: una línea con g++
g++ -O2 -std=c++17 ProyectoVision.cpp -o vision.bin $(pkg-config --cflags --libs opencv4)

# Opción 2: CMake (si tu OpenCV está en otra ruta:  cmake -DOpenCV_DIR=/ruta/lib/cmake/opencv4 ..)
mkdir build && cd build && cmake .. && make -j4

./vision.bin                       # demo con la cámara 0  (g: CPU/GPU, ESC: salir)
./vision.bin 1                     # cámara 1
./vision.bin datos/Imagen1.jpeg    # imagen fija
./vision.bin --pruebas datos/Imagen1.jpeg   # pruebas automáticas, sin ventanas
```

Si OpenCV ignora tu GPU (pasa con algunos dispositivos), se puede forzar con la variable `OPENCV_OPENCL_DEVICE`, por ejemplo `":GPU:"`.

---

## 9. Preguntas probables en la sustentación

1. **¿Qué diferencia hay entre correlación y convolución?** La convolución voltea el kernel. `filter2D` hace correlación; con kernels simétricos (Gauss) es igual; con Sobel solo cambia el signo.
2. **¿Por qué normalizar el kernel Gaussiano?** Para que `Σ K = 1` y el brillo medio no cambie.
3. **¿Por qué `σ` y `ksize` deben ir de la mano?** El kernel trunca la gaussiana en `±r`; si `σ` es grande y `r` pequeño, se corta una parte significativa de la campana (kernel casi uniforme); si `σ` es pequeño y `r` grande, los extremos valen ~0 y se desperdicia cómputo.
4. **¿Por qué Sobel usa un vector `[1 2 1]`?** Suaviza en la dirección perpendicular a la derivada para reducir el ruido que amplifica la derivada.
5. **¿Por qué `CV_32F` en Sobel?** Para conservar los gradientes negativos.
6. **¿Por qué se ecualiza el canal L y no B, G, R?** Para no alterar la cromaticidad.
7. **¿Qué hace el `clipLimit` de CLAHE?** Limita la altura de las barras del histograma de cada bloque y reparte el exceso, lo que limita la amplificación del ruido.
8. **¿Erosión vs dilatación? ¿Apertura vs cierre?** (mín vs máx; mín→máx elimina objetos claros pequeños, máx→mín rellena huecos oscuros pequeños).
9. **¿Por qué la morfología no es una convolución?** Usa mínimo/máximo, que no es una suma ponderada; no es lineal.
10. **¿Por qué `ocl::finish()` al medir en GPU?** Porque las llamadas OpenCL son asíncronas.
11. **¿Qué es la T-API?** `UMat` + `InputArray`: el mismo código se despacha a un kernel OpenCL o a la CPU según el tipo y `useOpenCL()`.

---

## 10. Referencias

- Material del curso, *Ejemplos de Código Unidad 1 y Unidad 2* (Visión Artificial, UPS): Ej. 3 (HSV/Lab), Ej. 4 (equalizeHist), Ej. 5 (CLAHE, medianBlur), Ej. 6 (GaussianBlur + Sobel), Ej. 7 (morfología), Ej. 8 (Canny), Ej. 9 (ruido sal y pimienta).
- OpenCV 4.x, documentación oficial: `filter2D`, `getDerivKernels`, `Sobel`, `GaussianBlur` (regla de σ por defecto), `equalizeHist`, `createCLAHE`, `morphologyEx`, `getStructuringElement`, y la guía de la T-API / `cv::UMat` — https://docs.opencv.org/4.x/
- R. C. González y R. E. Woods, *Digital Image Processing* (suavizado espacial, gradiente, ecualización de histograma, morfología).
- K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", *Graphics Gems IV*, 1994.
- I. Sobel y G. Feldman, "A 3×3 Isotropic Gradient Operator for Image Processing", 1968.
