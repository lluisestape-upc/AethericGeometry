import cv2
import mediapipe as mp
import numpy as np

# Inicialización estándar de MediaPipe
mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

manos = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7,
    max_num_hands=2
)

camara = cv2.VideoCapture(0)

# ==========================================
# --- CONFIGURACIÓN DE UX Y UMBRALES ---
# ==========================================
UMBRAL_PINZA = 60            # General para activar/desactivar vértices individuales
UMBRAL_PINZAS_JUNTAS = 90    # Distancia entre pinzas para activar cuadrado
UMBRAL_PIRAMIDE = 70         # Distancia permisiva entre puntas de los dedos para el gesto total
FRAME_COOLDOWN = 15          # Cuántos frames aguanta el instrumento sin manos antes de Reset
# ==========================================

# Variables de Estado de la Máquina
estado_actual = "IDLE" 
previo_doble_pinza = False
accion_pendiente_release = "NINGUNA"
contador_frames_sin_manos = 0

# Variables del Sintetizador (Visuales)
forma_onda = "SIN" # SIN, SAW, SQUARE, TRIANGLE
# Índices de vértices iniciales
dedos_activos_izq = [8]
dedos_activos_der = [8]

# Funciones auxiliares geométricas
def obtener_coords(mano, id_punto, ancho, alto):
    return int(mano.landmark[id_punto].x * ancho), int(mano.landmark[id_punto].y * alto)

def distancia(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

# Función auxiliar para contar dedos levantados (mano izquierda)
def contar_dedos_arriba(mano, alto):
    fingers = []
    for tip, knuckle in [(8,6), (12,10), (16,14), (20,18)]:
        if mano.landmark[tip].y < mano.landmark[knuckle].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

while True:
    exito, frame = camara.read()
    if not exito: break

    frame = cv2.flip(frame, 1) # Espejo
    alto, ancho, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = manos.process(frame_rgb)

    # Colectar manos detectadas
    manos_detectadas = []
    if resultados.multi_hand_landmarks:
        manos_detectadas = resultados.multi_hand_landmarks

    mano_izq_detectada = None
    mano_der_detectada = None

    # Asignación dinámica I/D
    for m in manos_detectadas:
        if m.landmark[4].x < m.landmark[17].x:
            mano_der_detectada = m
        else:
            mano_izq_detectada = m

    # ==========================================
    # LÓGICA DE CONTROL DEL SINTETIZADOR (Gated invisible)
    # ==========================================
    # Solo funciona si estamos en IDLE, pero visualmente ya no dice "Locked"
    if estado_actual == "IDLE":
        if mano_izq_detectada:
            dedos_num = contar_dedos_arriba(mano_izq_detectada, alto)
            if dedos_num == 1: forma_onda = "SIN"
            elif dedos_num == 2: forma_onda = "SAW"
            elif dedos_num == 3: forma_onda = "SQUARE"
            elif dedos_num == 4: forma_onda = "TRIANGLE"
    # ==========================================

    # ==========================================
    # LÓGICA DE CREACIÓN DE FORMAS
    # ==========================================
    if len(manos_detectadas) == 2 and mano_izq_detectada and mano_der_detectada:
        contador_frames_sin_manos = 0
        
        # Coordenadas críticas base
        p_izq = obtener_coords(mano_izq_detectada, 4, ancho, alto)
        i_izq = obtener_coords(mano_izq_detectada, 8, ancho, alto)
        p_der = obtener_coords(mano_der_detectada, 4, ancho, alto)
        i_der = obtener_coords(mano_der_detectada, 8, ancho, alto)

        # Evaluar Entradas Gestuales Básicas
        pinza_izq_ativa = distancia(p_izq, i_izq) < UMBRAL_PINZA
        pinza_der_activa = distancia(p_der, i_der) < UMBRAL_PINZA
        doble_pinza = pinza_izq_ativa and pinza_der_activa
        
        flanco_subida_pinza = doble_pinza and not previo_doble_pinza
        flanco_bajada_pinza = not doble_pinza and previo_doble_pinza

        # Gesto A: Beso de Pinzas -> Trigger Cuadrado
        mid_izq = (int((p_izq[0] + i_izq[0]) / 2), int((p_izq[1] + i_izq[1]) / 2))
        mid_der = (int((p_der[0] + i_der[0]) / 2), int((p_der[1] + i_der[1]) / 2))
        beso_pinzas = doble_pinza and (distancia(mid_izq, mid_der) < UMBRAL_PINZAS_JUNTAS)

        # Gesto B: "Pirámide" (Todos los dedos tocándose)
        dedos_tocandose = 0
        for id_dedo in [4, 8, 12, 16, 20]:
            c_izq = obtener_coords(mano_izq_detectada, id_dedo, ancho, alto)
            c_der = obtener_coords(mano_der_detectada, id_dedo, ancho, alto)
            if distancia(c_izq, c_der) < UMBRAL_PIRAMIDE:
                dedos_tocandose += 1
        
        # Si al menos 4 de los 5 pares de dedos se están tocando (permite un pequeño margen de error a la cámara)
        gesto_piramide = dedos_tocandose >= 4

        # --- Máquina de Estados (Transiciones) ---
        if estado_actual == "IDLE":
            if flanco_subida_pinza:
                estado_actual = "HILO"
                accion_pendiente_release = "NINGUNA"

        elif estado_actual == "HILO":
            if beso_pinzas: 
                estado_actual = "POLIGONO"
                dedos_activos_izq = [8] 
                dedos_activos_der = [8]
            elif gesto_piramide: # NUEVO TRIGGER TOTAL
                estado_actual = "POLIGONO"
                dedos_activos_izq = [8, 12, 16, 20] 
                dedos_activos_der = [8, 12, 16, 20]
            elif flanco_subida_pinza: 
                accion_pendiente_release = "RESET_IDLE"
            elif flanco_bajada_pinza:
                if accion_pendiente_release == "RESET_IDLE":
                    estado_actual = "IDLE"
                    dedos_activos_izq = [8] 
                    dedos_activos_der = [8]
                accion_pendiente_release = "NINGUNA"

        elif estado_actual == "POLIGONO":
            if beso_pinzas: 
                estado_actual = "HILO"
            
            # Añadir vértices individuales (Medio=12, Anular=16, Meñique=20)
            for dedo in [12, 16, 20]:
                coords_d_izq = obtener_coords(mano_izq_detectada, dedo, ancho, alto)
                if distancia(p_izq, coords_d_izq) < UMBRAL_PINZA and dedo not in dedos_activos_izq:
                    dedos_activos_izq.append(dedo)
                
                coords_d_der = obtener_coords(mano_der_detectada, dedo, ancho, alto)
                if distancia(p_der, coords_d_der) < UMBRAL_PINZA and dedo not in dedos_activos_der:
                    dedos_activos_der.append(dedo)

        previo_doble_pinza = doble_pinza

    else:
        # Lógica de COOLDOWN
        if estado_actual != "IDLE":
            contador_frames_sin_manos += 1
            if contador_frames_sin_manos > FRAME_COOLDOWN:
                estado_actual = "IDLE"
                previo_doble_pinza = False
                dedos_activos_izq = [8]
                dedos_activos_der = [8]
            else:
                cv2.rectangle(frame, (0,0), (ancho,alto), (0,0,255), 10)

    # ==========================================
    # --- DIBUJADO Y UI BASADO EN ESTADO ---
    # ==========================================
    color_sint = (0, 255, 0) if estado_actual == "HILO" else (0, 255, 255)
    pts_poligono = [] 

    if estado_actual == "HILO":
        if mano_izq_detectada and mano_der_detectada:
            i_izq_f = obtener_coords(mano_izq_detectada, 8, ancho, alto)
            i_der_f = obtener_coords(mano_der_detectada, 8, ancho, alto)
            cv2.line(frame, i_izq_f, i_der_f, color_sint, 4)
            cv2.circle(frame, i_izq_f, 8, color_sint, cv2.FILLED)
            cv2.circle(frame, i_der_f, 8, color_sint, cv2.FILLED)

    elif estado_actual == "POLIGONO":
        if mano_izq_detectada and mano_der_detectada:
            for d in [4, 8, 12, 16, 20]: 
                if d == 4 or d in dedos_activos_izq:
                    pts_poligono.append(obtener_coords(mano_izq_detectada, d, ancho, alto))
            for d in [20, 16, 12, 8, 4]: 
                if d == 4 or d in dedos_activos_der:
                    pts_poligono.append(obtener_coords(mano_der_detectada, d, ancho, alto))

            if len(pts_poligono) >= 4:
                matriz_poligono = np.array(pts_poligono, np.int32)
                cv2.polylines(frame, [matriz_poligono], isClosed=True, color=color_sint, thickness=4)
                for pt in pts_poligono:
                    cv2.circle(frame, pt, 6, (0, 200, 255), cv2.FILLED)

    # --- UI Y FEEDBACK VISUAL (Limpia) ---
    color_wave_ui = (255, 100, 255) # Color unificado para la UI del sintetizador

    if estado_actual != "IDLE":
        txt_vertices = len(pts_poligono) if estado_actual == "POLIGONO" else 2
        cv2.putText(frame, f"MODO: {estado_actual} ({txt_vertices} vertices)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_sint, 2)
    else:
        cv2.putText(frame, "IDLE - Haz doble pinza para iniciar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

    # Visualizador de Onda (Siempre se ve igual, sin texto de "Locked")
    cv2.putText(frame, f"SINT: {forma_onda} WAVE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_wave_ui, 2)
    
    start_x = 350
    cv2.rectangle(frame, (start_x-10, 60), (start_x+100, 120), color_wave_ui, 2)
    if forma_onda == "SIN": cv2.putText(frame, "~", (start_x+10, 105), cv2.FONT_HERSHEY_SIMPLEX, 2, color_wave_ui, 3)
    elif forma_onda == "SAW": cv2.putText(frame, "/|", (start_x+10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_wave_ui, 3)
    elif forma_onda == "SQUARE": cv2.putText(frame, "[]", (start_x+10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_wave_ui, 3)
    elif forma_onda == "TRIANGLE": cv2.putText(frame, "/\\", (start_x+10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_wave_ui, 3)

    # Dibujo de malla fina de fondo
    if resultados.multi_hand_landmarks:
        for mano_p in resultados.multi_hand_landmarks:
            mp_dibujo.draw_landmarks(frame, mano_p, mp_hands.HAND_CONNECTIONS, mp_dibujo.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=1))

    cv2.imshow("Aetheric Geometry", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camara.release()
cv2.destroyAllWindows()