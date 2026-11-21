// ============================================================================
//  ProyectoVision.cpp  -  Proyecto Integrador de Visión Artificial, PARTE I
//  (UN SOLO ARCHIVO: filtros + demo interactiva + pruebas automáticas)
//
//  Cuatro filtros vistos en clase, cada uno con versión CPU (cv::Mat) y GPU (cv::UMat):
//    1. Gaussiano            (convolución, pasa-bajos)            -> Ejemplos 6, 7, 8, 9
//    2. Sobel                (convolución, gradiente / bordes)    -> Ejemplos 6, 7, 8
//    3. Ecualización / CLAHE (histograma, canal L de CIELab)      -> Ejemplos 3, 4, 5
//    4. Morfológicos         (erosión, dilatación, apertura...)   -> Ejemplo 7
//  (El 5.º filtro, el artístico propio, se agregará combinando estos.)
//
//  COMPILAR (Linux / macOS con pkg-config):
//      g++ -O2 -std=c++17 ProyectoVision.cpp -o vision.bin $(pkg-config --cflags --libs opencv4)
//  o con CMake:  mkdir build && cd build && cmake .. && make
//
//  USO:
//      ./vision.bin                 demo con la cámara 0
//      ./vision.bin 1               demo con la cámara 1
//      ./vision.bin foto.jpg        demo con una imagen fija
//      ./vision.bin video.mp4       demo con un archivo de vídeo
//      ./vision.bin --pruebas [img] pruebas automáticas (sin ventanas)
//  Teclas en la demo:  g = alternar CPU (Mat) / GPU (UMat+OpenCL),  ESC = salir
//
//  Explicación matemática de cada filtro: docs/Explicacion_Filtros.md
//  Código generado con asistencia de IA (Claude) y adaptado; estructura de ventana y
//  parámetros basados en los ejemplos de código de las Unidades 1 y 2 del curso.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

// ############################################################################
//  PARTE A - FILTROS  (misma lógica para cv::Mat y cv::UMat mediante plantillas)
//  template<class M>: M = cv::Mat -> CPU;  M = cv::UMat -> T-API (OpenCL si está activo)
//  Esta parte no usa ventanas: se puede copiar tal cual al código nativo de Android.
// ############################################################################
namespace filtros {

// --------------------------- Parámetros de cada filtro ----------------------

struct ParamsGaussiano {
    int    ksize = 5;     // lado (impar) del kernel cuadrado ksize x ksize
    double sigma = 0.0;   // desviación estándar; si es <= 0 se deriva de ksize
};

struct ParamsSobel {
    int    ksize    = 3;    // 3, 5, 7, ... (impar >= 3): tamaño del kernel de derivada
    double ganancia = 2.0;  // factor de visualización de la magnitud del gradiente
};

enum class ModoEcualizacion { GLOBAL, CLAHE };

struct ParamsEcualizacion {
    ModoEcualizacion modo      = ModoEcualizacion::CLAHE;
    double           clipLimit = 2.0;  // CLAHE: límite de recorte del histograma por bloque
    int              tiles     = 8;    // CLAHE: rejilla tiles x tiles
};

enum class OpMorfologica { EROSION, DILATACION, APERTURA, CIERRE, GRADIENTE, TOPHAT, BLACKHAT };

struct ParamsMorfologia {
    OpMorfologica op          = OpMorfologica::GRADIENTE;
    int           ksize       = 3;                 // lado (impar) del elemento estructurante
    int           forma       = cv::MORPH_ELLIPSE; // MORPH_RECT, MORPH_CROSS o MORPH_ELLIPSE
    int           iteraciones = 1;
};

// --------------------------- Utilidades de kernels (CPU) --------------------
// Son públicas para poder imprimirlas / graficarlas en el informe y justificarlas.

// Fuerza un tamaño de kernel impar y >= minimo (un kernel de convolución necesita centro).
int forzarImpar(int k, int minimo = 1);

// Sigma por defecto cuando el usuario no lo da (misma regla que documenta OpenCV).
double sigmaPorDefecto(int ksize);

// Kernel Gaussiano 2D normalizado (suma = 1), CV_32F, tamaño ksize x ksize.
cv::Mat kernelGaussiano(int ksize, double sigma);

// Kernels 2D de Sobel (derivada en x e y) normalizados, CV_32F, ksize x ksize.
void kernelsSobel(int ksize, cv::Mat& kGx, cv::Mat& kGy);

const char* nombreOperacion(OpMorfologica op);

// --------------------------- Filtros (M = cv::Mat o cv::UMat) ---------------
// Entrada: CV_8UC1 o CV_8UC3 (BGR). `dst` se reserva automáticamente.
// No usar src y dst como la misma matriz.

// 1) Suavizado Gaussiano: dst = src (*) G   (convolución con filter2D).  Salida: igual que src.
template <class M> void gaussiano(const M& src, M& dst, const ParamsGaussiano& p);

// 2) Bordes de Sobel: dst = ganancia * sqrt(Gx^2 + Gy^2).  Salida: CV_8UC1.
template <class M> void sobel(const M& src, M& dst, const ParamsSobel& p);

// 3) Ecualización de histograma (global o CLAHE) sobre la luminancia L de CIELab.
//    Si src es de 1 canal se aplica directamente.  Salida: igual que src.
template <class M> void ecualizacion(const M& src, M& dst, const ParamsEcualizacion& p);

// 4) Operación morfológica con elemento estructurante ksize x ksize.  Salida: igual que src.
template <class M> void morfologia(const M& src, M& dst, const ParamsMorfologia& p);

}  // namespace filtros

namespace filtros {


// ============================================================================
//  Utilidades de kernels
// ============================================================================

int forzarImpar(int k, int minimo) {
    k = std::max(k, minimo);
    return (k % 2 == 0) ? k + 1 : k;  // par -> siguiente impar (el kernel necesita un píxel central)
}

double sigmaPorDefecto(int ksize) {
    // Regla documentada por OpenCV (cv::getGaussianKernel): el "ancho útil" de la campana
    // crece con el tamaño del kernel para que los bordes del kernel valgan ~0.
    return 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8;
}

// G(x,y) = exp( -(x^2 + y^2) / (2 sigma^2) ),   x,y en [-r, r],  r = ksize/2
// Se normaliza dividiendo por la suma de todos los coeficientes, de modo que
// sum(G) = 1: el filtro NO cambia el brillo medio de la imagen.
// (La constante 1/(2*pi*sigma^2) de la gaussiana continua se cancela al normalizar.)
cv::Mat kernelGaussiano(int ksize, double sigma) {
    ksize = forzarImpar(ksize);
    if (sigma <= 0.0) sigma = sigmaPorDefecto(ksize);

    const int r = ksize / 2;
    const double dosSigma2 = 2.0 * sigma * sigma;

    cv::Mat k(ksize, ksize, CV_32F);
    double suma = 0.0;
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {
            const double v = std::exp(-(x * x + y * y) / dosSigma2);
            k.at<float>(y + r, x + r) = static_cast<float>(v);
            suma += v;
        }
    }
    k /= suma;
    return k;
}

// Sobel = derivada (en un eje) x suavizado binomial (en el otro eje).
//   Sobel 3x3 en x:   1/8 * [1 2 1]^T * [-1 0 1]
// getDerivKernels entrega las dos partes 1D; el kernel 2D es su producto exterior
// (columna * fila).  Con normalize=true el resultado es la derivada "real"
// (niveles de gris por píxel), independiente del tamaño del kernel.
void kernelsSobel(int ksize, cv::Mat& kGx, cv::Mat& kGy) {
    ksize = std::min(forzarImpar(ksize, 3), 31);  // getDerivKernels admite hasta 31

    cv::Mat kx, ky;                                     // vectores columna ksize x 1
    cv::getDerivKernels(kx, ky, 1, 0, ksize, true, CV_32F);
    kGx = ky * kx.t();                                  // suavizado vertical x derivada horizontal

    cv::getDerivKernels(kx, ky, 0, 1, ksize, true, CV_32F);
    kGy = ky * kx.t();                                  // derivada vertical x suavizado horizontal
}

const char* nombreOperacion(OpMorfologica op) {
    switch (op) {
        case OpMorfologica::EROSION:   return "Erosion";
        case OpMorfologica::DILATACION:return "Dilatacion";
        case OpMorfologica::APERTURA:  return "Apertura";
        case OpMorfologica::CIERRE:    return "Cierre";
        case OpMorfologica::GRADIENTE: return "Gradiente morf.";
        case OpMorfologica::TOPHAT:    return "Top-hat";
        case OpMorfologica::BLACKHAT:  return "Black-hat";
    }
    return "?";
}

// ============================================================================
//  1) Gaussiano
// ============================================================================
template <class M>
void gaussiano(const M& src, M& dst, const ParamsGaussiano& p) {
    CV_Assert(src.depth() == CV_8U && (src.channels() == 1 || src.channels() == 3));

    const cv::Mat kernel = kernelGaussiano(p.ksize, p.sigma);

    // filter2D calcula  dst(x,y) = sum_{i,j} K(i,j) * src(x+i, y+j)  para cada canal.
    // Con BORDER_REFLECT_101 los píxeles fuera de la imagen se reflejan (sin borde negro).
    cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
}

// ============================================================================
//  2) Sobel
// ============================================================================
template <class M>
void sobel(const M& src, M& dst, const ParamsSobel& p) {
    CV_Assert(src.depth() == CV_8U && (src.channels() == 1 || src.channels() == 3));

    // El gradiente se calcula sobre la intensidad (escala de grises).
    M gris;
    if (src.channels() == 3) cv::cvtColor(src, gris, cv::COLOR_BGR2GRAY);
    else                     gris = src;

    cv::Mat kGx, kGy;
    kernelsSobel(p.ksize, kGx, kGy);

    // CV_32F para no perder el signo del gradiente (con 8U los negativos se saturarían a 0).
    M gx, gy, magnitud;
    cv::filter2D(gris, gx, CV_32F, kGx, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
    cv::filter2D(gris, gy, CV_32F, kGy, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);

    // |grad I| = sqrt(Gx^2 + Gy^2): magnitud del gradiente (fuerza del borde).
    cv::magnitude(gx, gy, magnitud);

    // Escala a 8 bits con saturación:  dst = saturate_cast<uchar>(ganancia * magnitud)
    magnitud.convertTo(dst, CV_8U, p.ganancia);
}

// ============================================================================
//  3) Ecualización de histograma (global o CLAHE) en el canal L de CIELab
// ============================================================================
template <class M>
void ecualizacion(const M& src, M& dst, const ParamsEcualizacion& p) {
    CV_Assert(src.depth() == CV_8U && (src.channels() == 1 || src.channels() == 3));

    // Objeto CLAHE reutilizable (crearlo en cada frame sería un desperdicio).
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    auto ecualizarCanal = [&](const M& entrada, M& salida) {
        if (p.modo == ModoEcualizacion::GLOBAL) {
            cv::equalizeHist(entrada, salida);
        } else {
            const int t = std::max(1, p.tiles);
            clahe->setClipLimit(std::max(0.0, p.clipLimit));
            clahe->setTilesGridSize(cv::Size(t, t));
            clahe->apply(entrada, salida);
        }
    };

    if (src.channels() == 1) {
        ecualizarCanal(src, dst);
        return;
    }

    // Color: se ecualiza SOLO la luminancia. Si se ecualizara cada canal B,G,R por separado
    // cambiarían las proporciones entre canales y los colores se distorsionarían.
    M lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    std::vector<M> canales;
    cv::split(lab, canales);          // canales[0] = L, [1] = a, [2] = b

    M lEcualizada;
    ecualizarCanal(canales[0], lEcualizada);
    canales[0] = lEcualizada;

    cv::merge(canales, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}

// ============================================================================
//  4) Morfología
// ============================================================================
template <class M>
void morfologia(const M& src, M& dst, const ParamsMorfologia& p) {
    CV_Assert(src.depth() == CV_8U && (src.channels() == 1 || src.channels() == 3));

    int tipo = cv::MORPH_GRADIENT;
    switch (p.op) {
        case OpMorfologica::EROSION:    tipo = cv::MORPH_ERODE;    break;
        case OpMorfologica::DILATACION: tipo = cv::MORPH_DILATE;   break;
        case OpMorfologica::APERTURA:   tipo = cv::MORPH_OPEN;     break;
        case OpMorfologica::CIERRE:     tipo = cv::MORPH_CLOSE;    break;
        case OpMorfologica::GRADIENTE:  tipo = cv::MORPH_GRADIENT; break;
        case OpMorfologica::TOPHAT:     tipo = cv::MORPH_TOPHAT;   break;
        case OpMorfologica::BLACKHAT:   tipo = cv::MORPH_BLACKHAT; break;
    }

    const int k = forzarImpar(p.ksize);
    const cv::Mat elemento = cv::getStructuringElement(p.forma, cv::Size(k, k));

    // En imágenes de color la operación se aplica independientemente a cada canal.
    cv::morphologyEx(src, dst, tipo, elemento, cv::Point(-1, -1), std::max(1, p.iteraciones));
}

}  // namespace filtros

// ############################################################################
//  PARTE B - PRUEBAS AUTOMÁTICAS  (./vision.bin --pruebas)
//  A) matemática de kernels  B) propiedades  C) Mat vs UMat  D) tiempos
// ############################################################################
using namespace filtros;

static int fallos = 0;

static void check(bool ok, const string& nombre, const string& detalle = "") {
    cout << (ok ? "  [OK]    " : "  [FALLO] ") << nombre;
    if (!detalle.empty()) cout << "  (" << detalle << ")";
    cout << endl;
    if (!ok) ++fallos;
}

static double maxDiff(const Mat& a, const Mat& b) {
    Mat d;
    absdiff(a, b, d);
    d = d.reshape(1);          // todos los canales en una sola "capa"
    double mx = 0;
    minMaxLoc(d, nullptr, &mx);
    return mx;
}

static Mat descargar(const UMat& u) {
    Mat m;
    u.copyTo(m);
    return m;
}

// Mediana de `n` ejecuciones (ms) tras unas ejecuciones de calentamiento
// (la primera llamada OpenCL compila los kernels y distorsiona la medida).
template <class F>
static double medianaMs(F&& accion, bool gpu, int n = 15, int calentamiento = 3) {
    for (int i = 0; i < calentamiento; ++i) { accion(); if (gpu) ocl::finish(); }
    vector<double> t;
    for (int i = 0; i < n; ++i) {
        auto t0 = chrono::high_resolution_clock::now();
        accion();
        if (gpu) ocl::finish();
        auto t1 = chrono::high_resolution_clock::now();
        t.push_back(chrono::duration<double, milli>(t1 - t0).count());
    }
    nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}

static int ejecutarPruebas(int argc, char** argv) {   // argv[1] = ruta de imagen (opcional)
    const string ruta = (argc > 1) ? argv[1] : "../datos/Imagen1.jpeg";
    Mat img = imread(ruta, IMREAD_COLOR);
    if (img.empty()) { cerr << "No se pudo leer la imagen: " << ruta << endl; return 2; }

    cout << "OpenCV " << CV_VERSION << " | imagen " << img.cols << "x" << img.rows << endl;
    ocl::setUseOpenCL(true);   // useOpenCL() es true solo si OpenCV pudo crear un contexto real
    cout << "OpenCL utilizable: " << (ocl::useOpenCL() ? "SI" : "NO");
    if (ocl::useOpenCL()) cout << " | dispositivo: " << ocl::Device::getDefault().name();
    cout << endl;

    Mat gris;
    cvtColor(img, gris, COLOR_BGR2GRAY);

    // ------------------------------------------------------------------ A
    cout << "\nA) Matemática de los kernels" << endl;
    {
        for (int k : {3, 5, 9, 15}) {
            Mat g = kernelGaussiano(k, 0);
            check(abs(sum(g)[0] - 1.0) < 1e-6, "Gauss " + to_string(k) + "x" + to_string(k) + ": suma = 1");
            Mat gT, gF;
            transpose(g, gT);
            flip(g, gF, -1);
            check(maxDiff(g, gT) == 0 && maxDiff(g, gF) == 0, "Gauss " + to_string(k) + "x" + to_string(k) + ": simétrico");
        }
        Mat g5 = kernelGaussiano(5, 1.0);
        double mx; Point pmx;
        minMaxLoc(g5, nullptr, &mx, nullptr, &pmx);
        check(pmx == Point(2, 2), "Gauss 5x5: el máximo está en el centro");

        // Sobel 3x3 conocido (escalado 1/8 para que sea la derivada real).
        Mat kGx, kGy;
        kernelsSobel(3, kGx, kGy);
        Mat esperadoX = (Mat_<float>(3, 3) << -1.f, 0.f, 1.f, -2.f, 0.f, 2.f, -1.f, 0.f, 1.f) / 8.0;
        Mat esperadoY = (Mat_<float>(3, 3) << -1.f, -2.f, -1.f, 0.f, 0.f, 0.f, 1.f, 2.f, 1.f) / 8.0;
        check(norm(kGx, esperadoX, NORM_INF) < 1e-7, "Sobel 3x3 Gx == [-1 0 1; -2 0 2; -1 0 1]/8");
        check(norm(kGy, esperadoY, NORM_INF) < 1e-7, "Sobel 3x3 Gy == [-1 -2 -1; 0 0 0; 1 2 1]/8");

        for (int k : {3, 5, 7}) {
            kernelsSobel(k, kGx, kGy);
            Mat miX, miY, refX, refY;
            filter2D(gris, miX, CV_32F, kGx, Point(-1, -1), 0, BORDER_REFLECT_101);
            filter2D(gris, miY, CV_32F, kGy, Point(-1, -1), 0, BORDER_REFLECT_101);
            const double escala = 1.0 / double(1 << (2 * k - 3));   // normalización de cv::Sobel
            Sobel(gris, refX, CV_32F, 1, 0, k, escala);
            Sobel(gris, refY, CV_32F, 0, 1, k, escala);
            check(norm(miX, refX, NORM_INF) < 1e-3 && norm(miY, refY, NORM_INF) < 1e-3,
                  "Sobel " + to_string(k) + "x" + to_string(k) + " via filter2D == cv::Sobel");
        }

        // Gaussiano propio vs cv::GaussianBlur (usa aritmética de punto fijo -> tolerancia 2 niveles)
        for (int k : {3, 5, 9, 15}) {
            Mat mio, ref;
            gaussiano(img, mio, ParamsGaussiano{k, 0.0});
            GaussianBlur(img, ref, Size(k, k), sigmaPorDefecto(k), sigmaPorDefecto(k), BORDER_REFLECT_101);
            const double d = maxDiff(mio, ref);
            check(d <= 2, "Gaussiano " + to_string(k) + "x" + to_string(k) + " == cv::GaussianBlur", "dif max " + to_string((int)d));
        }
    }

    // ------------------------------------------------------------------ B
    cout << "\nB) Propiedades de cada filtro" << endl;
    {
        // Gaussiano conserva el brillo medio y reduce la varianza (suaviza).
        Mat g; gaussiano(gris, g, ParamsGaussiano{9, 0.0});
        Scalar mO, sO, mG, sG;
        meanStdDev(gris, mO, sO); meanStdDev(g, mG, sG);
        check(abs(mO[0] - mG[0]) < 1.0, "Gaussiano conserva el brillo medio", "dif " + to_string(abs(mO[0] - mG[0])));
        check(sG[0] < sO[0], "Gaussiano reduce la desviación estándar (suaviza)");

        // Sobel: imagen constante -> sin bordes; escalón vertical -> borde solo en el escalón.
        Mat plano(100, 100, CV_8UC1, Scalar(120)), sp;
        sobel(plano, sp, ParamsSobel{3, 2.0});
        check(countNonZero(sp) == 0, "Sobel de imagen constante = 0");
        Mat escalon(100, 100, CV_8UC1, Scalar(0));
        escalon(Rect(50, 0, 50, 100)).setTo(255);
        Mat se;
        sobel(escalon, se, ParamsSobel{3, 1.0});
        check(se.at<uchar>(50, 49) > 100 && se.at<uchar>(50, 50) > 100 && se.at<uchar>(50, 10) == 0 && se.at<uchar>(50, 90) == 0,
              "Sobel detecta solo el escalón vertical");

        // Ecualización: sobre una imagen de bajo contraste debe AUMENTAR el contraste.
        Mat bajo;
        gris.convertTo(bajo, CV_8U, 0.25, 100);               // rango comprimido a [100, ~164]
        Mat bajoBGR; cvtColor(bajo, bajoBGR, COLOR_GRAY2BGR);
        Mat eg, ec;
        ecualizacion(bajo, eg, ParamsEcualizacion{ModoEcualizacion::GLOBAL, 2.0, 8});
        ecualizacion(bajoBGR, ec, ParamsEcualizacion{ModoEcualizacion::CLAHE, 4.0, 8});
        Scalar m0, s0, m1, s1, m2, s2;
        meanStdDev(bajo, m0, s0); meanStdDev(eg, m1, s1);
        Mat ecGris; cvtColor(ec, ecGris, COLOR_BGR2GRAY);
        meanStdDev(ecGris, m2, s2);
        check(s1[0] > 2 * s0[0], "Ecualización global aumenta el contraste", to_string(s0[0]) + " -> " + to_string(s1[0]));
        check(s2[0] > 2 * s0[0], "CLAHE (color, canal L) aumenta el contraste", to_string(s0[0]) + " -> " + to_string(s2[0]));
        check(ec.type() == CV_8UC3 && ec.size() == bajoBGR.size(), "Ecualización color conserva tipo y tamaño");

        // Morfología: erosión <= original <= dilatación;  apertura <= original <= cierre.
        Mat er, di, ap, ci, difA, difB;
        ParamsMorfologia pm; pm.ksize = 5;
        pm.op = OpMorfologica::EROSION;    morfologia(gris, er, pm);
        pm.op = OpMorfologica::DILATACION; morfologia(gris, di, pm);
        pm.op = OpMorfologica::APERTURA;   morfologia(gris, ap, pm);
        pm.op = OpMorfologica::CIERRE;     morfologia(gris, ci, pm);
        check(countNonZero(er > gris) == 0 && countNonZero(di < gris) == 0, "Erosión <= original <= dilatación");
        check(countNonZero(ap > gris) == 0 && countNonZero(ci < gris) == 0, "Apertura <= original <= cierre");
        Mat grad, esperado;
        pm.op = OpMorfologica::GRADIENTE; morfologia(gris, grad, pm);
        subtract(di, er, esperado);
        check(maxDiff(grad, esperado) == 0, "Gradiente morfológico == dilatación - erosión");
        // Idempotencia: abrir dos veces = abrir una vez
        Mat ap2; morfologia(ap, ap2, ParamsMorfologia{OpMorfologica::APERTURA, 5, MORPH_ELLIPSE, 1});
        check(maxDiff(ap, ap2) == 0, "La apertura es idempotente");
    }

    // ------------------------------------------------------------------ C
    cout << "\nC) Equivalencia CPU (Mat) vs T-API (UMat)" << endl;
    {
        UMat uimg; img.copyTo(uimg);
        auto comparar = [&](const string& nombre, auto&& filtroMat, auto&& filtroUMat, double tol) {
            Mat salM; UMat salU;
            filtroMat(img, salM);
            filtroUMat(uimg, salU);
            Mat salUm = descargar(salU);
            const double d = maxDiff(salM, salUm);
            check(salM.size() == salUm.size() && salM.type() == salUm.type() && d <= tol, nombre, "dif max " + to_string((int)d));
        };

        for (int k : {3, 9, 21}) {
            ParamsGaussiano p{k, 0.0};
            comparar("Gaussiano " + to_string(k), [&](const Mat& s, Mat& d) { gaussiano(s, d, p); },
                     [&](const UMat& s, UMat& d) { gaussiano(s, d, p); }, 2);
        }
        for (int k : {3, 5, 7}) {
            ParamsSobel p{k, 2.0};
            comparar("Sobel " + to_string(k), [&](const Mat& s, Mat& d) { sobel(s, d, p); },
                     [&](const UMat& s, UMat& d) { sobel(s, d, p); }, 2);
        }
        for (auto modo : {ModoEcualizacion::GLOBAL, ModoEcualizacion::CLAHE}) {
            ParamsEcualizacion p{modo, 3.0, 8};
            comparar(modo == ModoEcualizacion::GLOBAL ? "Ecualización global" : "CLAHE",
                     [&](const Mat& s, Mat& d) { ecualizacion(s, d, p); },
                     [&](const UMat& s, UMat& d) { ecualizacion(s, d, p); }, 3);
        }
        for (auto op : {OpMorfologica::EROSION, OpMorfologica::DILATACION, OpMorfologica::APERTURA,
                        OpMorfologica::CIERRE, OpMorfologica::GRADIENTE, OpMorfologica::TOPHAT, OpMorfologica::BLACKHAT}) {
            ParamsMorfologia p{op, 7, MORPH_ELLIPSE, 1};
            comparar(string("Morfología ") + nombreOperacion(op),
                     [&](const Mat& s, Mat& d) { morfologia(s, d, p); },
                     [&](const UMat& s, UMat& d) { morfologia(s, d, p); }, 0);
        }
    }

    // ------------------------------------------------------------------ D
    cout << "\nD) Tiempos (mediana, ms) sobre 1280x720" << endl;
    if (!ocl::useOpenCL())
        cout << "   AVISO: OpenCL no está activo; la columna UMat se ejecuta en CPU.\n"
                "   Estos tiempos NO sirven como comparación CPU vs GPU.\n";
    else if (ocl::Device::getDefault().type() == ocl::Device::TYPE_CPU)
        cout << "   AVISO: el dispositivo OpenCL es una CPU (emulación); no es una GPU real.\n"
                "   Estos tiempos NO sirven como comparación CPU vs GPU.\n";
    {
        Mat grande; resize(img, grande, Size(1280, 720));
        UMat ugrande; grande.copyTo(ugrande);
        const bool gpu = ocl::useOpenCL();

        cout << "   " << left << setw(26) << "Filtro" << right << setw(12) << "CPU Mat" << setw(12) << "UMat" << setw(10) << "Aceler." << endl;
        auto fila = [&](const string& nombre, auto&& fM, auto&& fU) {
            Mat dM; UMat dU;
            const double tc = medianaMs([&] { fM(grande, dM); }, false);
            const double tg = medianaMs([&] { fU(ugrande, dU); }, gpu);
            cout << "   " << left << setw(26) << nombre << right << fixed << setprecision(2)
                 << setw(12) << tc << setw(12) << tg << setw(9) << (tc / tg) << "x" << endl;
        };
        for (int k : {3, 9, 15, 31}) {
            ParamsGaussiano p{k, 0.0};
            fila("Gaussiano " + to_string(k) + "x" + to_string(k), [&](const Mat& s, Mat& d) { gaussiano(s, d, p); },
                 [&](const UMat& s, UMat& d) { gaussiano(s, d, p); });
        }
        for (int k : {3, 7}) {
            ParamsSobel p{k, 2.0};
            fila("Sobel " + to_string(k) + "x" + to_string(k), [&](const Mat& s, Mat& d) { sobel(s, d, p); },
                 [&](const UMat& s, UMat& d) { sobel(s, d, p); });
        }
        {
            ParamsEcualizacion p{ModoEcualizacion::CLAHE, 2.0, 8};
            fila("CLAHE (Lab)", [&](const Mat& s, Mat& d) { ecualizacion(s, d, p); },
                 [&](const UMat& s, UMat& d) { ecualizacion(s, d, p); });
        }
        for (int k : {3, 15, 31}) {
            ParamsMorfologia p{OpMorfologica::GRADIENTE, k, MORPH_ELLIPSE, 1};
            fila("Morf. gradiente " + to_string(k) + "x" + to_string(k), [&](const Mat& s, Mat& d) { morfologia(s, d, p); },
                 [&](const UMat& s, UMat& d) { morfologia(s, d, p); });
        }
    }

    cout << "\n" << (fallos == 0 ? "TODAS LAS PRUEBAS PASARON" : "HAY PRUEBAS FALLIDAS: " + to_string(fallos)) << endl;
    return fallos == 0 ? 0 : 1;
}

// ############################################################################
//  PARTE C - DEMO INTERACTIVA (ventana con trackbars, cámara / imagen / vídeo)
// ############################################################################

static const string VENTANA = "Parte I - 4 filtros (g: CPU/GPU, ESC: salir)";

// Mide en ms lo que tarda `accion`. En GPU las órdenes OpenCL son asíncronas, por eso
// se llama a ocl::finish() antes de parar el reloj (si no, mediríamos solo el "encolado").
template <class F>
static double medirMs(F&& accion, bool gpu) {
    auto t0 = chrono::high_resolution_clock::now();
    accion();
    if (gpu) ocl::finish();
    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(t1 - t0).count();
}

static void pegar(Mat& lienzo, const Mat& vista, int col, int fila) {
    Mat tmp = vista;
    if (tmp.channels() == 1) cvtColor(vista, tmp, COLOR_GRAY2BGR);
    tmp.copyTo(lienzo(Rect(col * vista.cols, fila * vista.rows, vista.cols, vista.rows)));
}

static void rotular(Mat& vista, const string& texto) {
    // Sombra negra + texto blanco: legible sobre cualquier fondo.
    putText(vista, texto, Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0), 3, LINE_AA);
    putText(vista, texto, Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 255), 1, LINE_AA);
}

static int pos(const string& nombre) { return getTrackbarPos(nombre, VENTANA); }

static int ejecutarDemo(int argc, char* argv[]) {
    // ---- Fuente de imagen: imagen fija, vídeo o cámara --------------------------------
    Mat imagenFija;
    VideoCapture video;
    if (argc > 1) {
        const string arg = argv[1];
        imagenFija = imread(arg, IMREAD_COLOR);
        if (imagenFija.empty()) {
            const bool esNumero = all_of(arg.begin(), arg.end(), ::isdigit);
            if (esNumero) video.open(stoi(arg)); else video.open(arg);
        }
    } else {
        video.open(0);
    }
    if (imagenFija.empty() && !video.isOpened()) {
        cerr << "No se pudo abrir la fuente de imagen." << endl;
        return 1;
    }

    // haveOpenCL() solo dice que existe la biblioteca; useOpenCL() es true únicamente si OpenCV
    // logró crear un contexto con un dispositivo utilizable. Probamos activándolo.
    ocl::setUseOpenCL(true);
    const bool hayOpenCL = ocl::useOpenCL();
    bool usarGPU = false;                 // arrancamos en CPU; 'g' alterna
    ocl::setUseOpenCL(false);

    // ---- Ventana y controles (trackbars) ------------------------------------------------
    namedWindow(VENTANA, WINDOW_AUTOSIZE);
    auto crear = [&](const string& n, int max, int ini) {
        createTrackbar(n, VENTANA, nullptr, max, nullptr, nullptr);
        setTrackbarPos(n, VENTANA, ini);
    };
    crear("Gauss kernel", 61, 5);
    crear("Gauss sigma x10", 200, 0);      // 0 -> sigma automático
    crear("Sobel kernel", 31, 3);
    crear("Sobel ganancia x10", 100, 20);
    crear("Ecual 0=Global 1=CLAHE", 1, 1);
    crear("CLAHE clip x10", 100, 20);
    crear("CLAHE tiles", 16, 8);
    crear("Morf op 0-6", 6, 4);            // 0 ero,1 dil,2 apert,3 cierre,4 grad,5 tophat,6 blackhat
    crear("Morf kernel", 61, 3);
    crear("Morf forma 0=R 1=X 2=E", 2, 2);

    Mat lienzo;
    double fps = 0.0;

    while (3 == 3) {
        auto tFrame0 = chrono::high_resolution_clock::now();

        // ---- Captura y normalización de tamaño (ancho fijo -> lienzo de tamaño constante) -------
        Mat frame;
        if (!imagenFija.empty()) frame = imagenFija.clone();
        else { video >> frame; if (frame.empty()) break; }
        const double escala = 480.0 / frame.cols;
        resize(frame, frame, Size(), escala, escala, INTER_AREA);
        if (imagenFija.empty()) flip(frame, frame, 1);   // efecto espejo solo con cámara/vídeo

        // ---- Lectura de parámetros -----------------------------------------------------
        ParamsGaussiano pg;  pg.ksize = pos("Gauss kernel");  pg.sigma = pos("Gauss sigma x10") / 10.0;
        ParamsSobel     ps;  ps.ksize = pos("Sobel kernel");  ps.ganancia = pos("Sobel ganancia x10") / 10.0;
        ParamsEcualizacion pe;
        pe.modo = pos("Ecual 0=Global 1=CLAHE") ? ModoEcualizacion::CLAHE : ModoEcualizacion::GLOBAL;
        pe.clipLimit = pos("CLAHE clip x10") / 10.0;
        pe.tiles = pos("CLAHE tiles");
        ParamsMorfologia pm;
        pm.op = static_cast<OpMorfologica>(pos("Morf op 0-6"));
        pm.ksize = pos("Morf kernel");
        const int formas[3] = {MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE};
        pm.forma = formas[min(2, pos("Morf forma 0=R 1=X 2=E"))];

        // ---- Aplicación de los 4 filtros en el pipeline elegido -------------------------------
        Mat sG, sS, sE, sM;
        double tG, tS, tE, tM, tSubida = 0.0, tBajada = 0.0;

        if (usarGPU) {
            // Pipeline GPU (T-API): subir el frame a UMat, filtrar, bajar resultados para mostrarlos.
            UMat uFrame, uG, uS, uE, uM;
            tSubida = medirMs([&] { frame.copyTo(uFrame); }, true);
            tG = medirMs([&] { gaussiano(uFrame, uG, pg); }, true);
            tS = medirMs([&] { sobel(uFrame, uS, ps); }, true);
            tE = medirMs([&] { ecualizacion(uFrame, uE, pe); }, true);
            tM = medirMs([&] { morfologia(uFrame, uM, pm); }, true);
            tBajada = medirMs([&] { uG.copyTo(sG); uS.copyTo(sS); uE.copyTo(sE); uM.copyTo(sM); }, true);
        } else {
            // Pipeline CPU: matrices tradicionales cv::Mat.
            tG = medirMs([&] { gaussiano(frame, sG, pg); }, false);
            tS = medirMs([&] { sobel(frame, sS, ps); }, false);
            tE = medirMs([&] { ecualizacion(frame, sE, pe); }, false);
            tM = medirMs([&] { morfologia(frame, sM, pm); }, false);
        }

        // ---- Composición del lienzo 3x2 ----------------------------------------------------
        if (lienzo.empty() || lienzo.cols != frame.cols * 3 || lienzo.rows != frame.rows * 2)
            lienzo = Mat::zeros(Size(frame.cols * 3, frame.rows * 2), CV_8UC3);

        auto f2 = [](double v) { ostringstream s; s << fixed << setprecision(2) << v; return s.str(); };

        Mat vOrig = frame.clone(); rotular(vOrig, "Original");
        Mat vG = sG.clone(); if (vG.channels() == 1) cvtColor(vG, vG, COLOR_GRAY2BGR);
        rotular(vG, "Gaussiano " + to_string(forzarImpar(pg.ksize)) + "x" + to_string(forzarImpar(pg.ksize)) + "  " + f2(tG) + " ms");
        Mat vS; cvtColor(sS, vS, COLOR_GRAY2BGR);
        rotular(vS, "Sobel " + to_string(min(31, forzarImpar(ps.ksize, 3))) + "x" + to_string(min(31, forzarImpar(ps.ksize, 3))) + "  " + f2(tS) + " ms");
        Mat vE = sE.clone();
        rotular(vE, string(pe.modo == ModoEcualizacion::CLAHE ? "CLAHE (L Lab)" : "Ecualizacion global (L Lab)") + "  " + f2(tE) + " ms");
        Mat vM = sM.clone();
        rotular(vM, string(nombreOperacion(pm.op)) + " " + to_string(forzarImpar(pm.ksize)) + "x" + to_string(forzarImpar(pm.ksize)) + "  " + f2(tM) + " ms");

        pegar(lienzo, vOrig, 0, 0);
        pegar(lienzo, vG, 1, 0);
        pegar(lienzo, vS, 2, 0);
        pegar(lienzo, vE, 0, 1);
        pegar(lienzo, vM, 1, 1);

        // Panel de información
        Mat info = Mat::zeros(frame.size(), CV_8UC3);
        const double suma = tG + tS + tE + tM;
        vector<string> lineas = {
            string("Pipeline: ") + (usarGPU ? "GPU (cv::UMat + OpenCL)" : "CPU (cv::Mat)"),
            string("OpenCL: ") + (hayOpenCL ? ocl::Device::getDefault().name() : "no disponible"),
            "Suma 4 filtros: " + f2(suma) + " ms",
            usarGPU ? "Subida RAM->VRAM: " + f2(tSubida) + " ms" : "",
            usarGPU ? "Bajada VRAM->RAM: " + f2(tBajada) + " ms" : "",
            "FPS (bucle completo): " + f2(fps),
            "Frame: " + to_string(frame.cols) + "x" + to_string(frame.rows),
            "[g] CPU/GPU   [ESC] salir"};
        int y = 28;
        for (const auto& l : lineas) {
            if (!l.empty()) putText(info, l, Point(10, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
            y += 26;
        }
        pegar(lienzo, info, 2, 1);

        imshow(VENTANA, lienzo);

        const int tecla = waitKey(1);
        if (tecla == 27) break;
        if (tecla == 'g' || tecla == 'G') {
            if (hayOpenCL) { usarGPU = !usarGPU; ocl::setUseOpenCL(usarGPU); }
            else cout << "OpenCL no esta disponible en este equipo." << endl;
        }

        auto tFrame1 = chrono::high_resolution_clock::now();
        const double dt = chrono::duration<double>(tFrame1 - tFrame0).count();
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6));   // media móvil para que el número no parpadee
    }

    video.release();
    destroyAllWindows();
    return 0;
}


int main(int argc, char* argv[]) {
    if (argc > 1 && string(argv[1]) == "--pruebas") return ejecutarPruebas(argc - 1, argv + 1);
    return ejecutarDemo(argc, argv);
}
