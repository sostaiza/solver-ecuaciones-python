
"""
Programa de consola para resolver sistemas de ecuaciones lineales Ax = b
con menú mejorado. Incluye:
  1) Definir dimensiones m x n
  2) Ingresar matriz A
  3) Ingresar vector b
  4) Resolver Ax=b con decimales configurables
  5) Ver detalles numéricos y diagnóstico
  6) Estado actual (dimensiones, A, b)
  7) Cambiar número de decimales
  8) Cargar ejemplo rápido
  0) Salir

Soporta casos cuadrados, sobredeterminados y subdeterminados.
"""
from __future__ import annotations
import sys
from typing import Optional, Tuple

try:
    import numpy as np
except Exception:
    print("Este programa requiere la librería 'numpy'. Instálala con: pip install numpy")
    sys.exit(1)

# -------------------- Utilidades de entrada -------------------- #

def _to_float(s: str) -> float:
    s = s.strip().replace(',', '.')
    return float(s)


def leer_entero_positivo(prompt: str) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            print("Ingresa un entero positivo.")


def leer_matriz(m: int, n: int, nombre: str = "A") -> np.ndarray:
    print(f"\nIngresa la matriz {nombre} de tamaño {m}x{n}.")
    filas = []
    for i in range(m):
        while True:
            fila_str = input(f"Fila {i+1}: ")
            try:
                vals = [_to_float(x) for x in fila_str.replace('\t', ' ').split()]
                if len(vals) != n:
                    raise ValueError
                filas.append(vals)
                break
            except ValueError:
                print(f"Debe haber exactamente {n} valores en la fila {i+1}.")
    return np.array(filas, dtype=float)


def leer_vector(m: int, nombre: str = "b") -> np.ndarray:
    print(f"\nIngresa el vector columna {nombre} de tamaño {m}.")
    valores = []
    while len(valores) < m:
        linea = input(f"Valores faltantes {m - len(valores)}: ")
        try:
            valores.extend([_to_float(x) for x in linea.replace('\t', ' ').split()])
        except ValueError:
            print("Solo números, por favor.")
    return np.array(valores[:m], dtype=float).reshape(m, 1)


# -------------------- Núcleo de resolución -------------------- #

def estado_matriz(A: np.ndarray) -> dict:
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(m, n) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    rango = int((s > tol).sum())
    cond = (s[0] / s[-1]) if (s.size > 0 and s[-1] > 0) else np.inf
    return {"m": m, "n": n, "rank": rango, "cond": cond, "singular": rango < min(m, n)}


def resolver_sistema(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
    m, n = A.shape
    info = {"metodo": None, "mensaje": None, "residuo_norma2": None}
    est = estado_matriz(A)

    if m == n and est["rank"] == n:
        try:
            x = np.linalg.solve(A, b)
            r = A @ x - b
            info.update({"metodo": "solve", "mensaje": "Solución única", "residuo_norma2": float(np.linalg.norm(r))})
            return x, info
        except np.linalg.LinAlgError:
            pass

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = A @ x - b
    info.update({"metodo": "lstsq", "mensaje": "Solución por mínimos cuadrados", "residuo_norma2": float(np.linalg.norm(r))})
    return x, info


def formatear_vector(x: np.ndarray, decimales: int = 3) -> str:
    return "[" + ", ".join(f"{v:.{decimales}f}" for v in x.reshape(-1)) + "]"


# -------------------- Menú -------------------- #

def menu():
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    m = n = None
    decimales = 3

    acciones = {
        "1": "Definir dimensiones (m x n)",
        "2": "Ingresar/editar matriz A",
        "3": "Ingresar/editar vector b",
        "4": "Resolver Ax = b",
        "5": "Ver detalles numéricos",
        "6": "Ver estado actual",
        "7": "Cambiar decimales de salida",
        "8": "Cargar ejemplo rápido",
        "0": "Salir"
    }

    while True:
        print("\n===== MENÚ =====")
        for k in sorted(acciones.keys()):
            print(f" {k}) {acciones[k]}")
        op = input("Elige una opción: ").strip()

        if op == "1":
            m = leer_entero_positivo("m: ")
            n = leer_entero_positivo("n: ")
            A = b = None
            print(f"Dimensiones {m}x{n} establecidas.")

        elif op == "2":
            if m is None or n is None:
                print("Define dimensiones primero.")
                continue
            A = leer_matriz(m, n)
            print("Matriz A guardada.")

        elif op == "3":
            if m is None:
                print("Define dimensiones primero.")
                continue
            b = leer_vector(m)
            print("Vector b guardado.")

        elif op == "4":
            if A is None or b is None:
                print("Falta A o b.")
                continue
            if b.shape[0] != A.shape[0]:
                print("Dimensiones incompatibles.")
                continue
            x, info = resolver_sistema(A, b)
            print("\n===== RESULTADO =====")
            print(info["mensaje"])
            print("x =", formatear_vector(x, decimales))
            print(f"‖Ax−b‖₂ = {info['residuo_norma2']:.3e}")

        elif op == "5":
            if A is None:
                print("Ingresa A.")
                continue
            est = estado_matriz(A)
            print("\n===== DETALLES =====")
            print(f"Tamaño: {est['m']}x{est['n']}")
            print(f"Rango: {est['rank']}")
            print(f"Condición κ: {est['cond']:.3e}")
            if est['singular']:
                print("A es singular o mal condicionada.")

        elif op == "6":
            print("\n===== ESTADO ACTUAL =====")
            print(f"Dimensiones: {m}x{n}" if m else "Sin definir")
            print(f"Matriz A: {'definida' if A is not None else 'no definida'}")
            print(f"Vector b: {'definido' if b is not None else 'no definido'}")
            print(f"Decimales de salida: {decimales}")

        elif op == "7":
            decimales = leer_entero_positivo("Número de decimales: ")
            print(f"Decimales fijados a {decimales}.")

        elif op == "8":
            m, n = 2, 2
            A = np.array([[2.0, 1.0], [5.0, 7.0]])
            b = np.array([[11.0], [13.0]])
            print("Ejemplo cargado: 2x2 con solución única.")

        elif op == "0":
            print("Adiós")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
