import os
import cv2
import torch
import ffmpeg
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# CONFIGURACIÓN DE PARÁMETROS
# ==============================================================================
VIDEO_PATH = "videos/GX011259_SHORT.mp4"  # Ruta del video original
OUTPUT_GRID_PATH = "final_comparison_grid.mp4"  # Ruta de la cuadrícula final

# Diccionario con los 6 modelos a comparar
MODEL_CONFIGS = {
    "YOLOv26n": "runs/detect/train/weights/best.pt",
    "YOLOv26s": "runs/detect/train-2/weights/best.pt",
    "YOLOv26s_Custom": "runs/detect/train-3/weights/best.pt",
    "YOLOv8n": "runs/detect/train-4/weights/best.pt",
    "YOLOv11n": "runs/detect/train-5/weights/best.pt",
    "YOLOv26s_GDL": "runs/detect/train-7/weights/best.pt"
}

FRAME_STRIDE = 3          # 1 = procesar cada frame, 2 = procesar uno de cada dos, etc.
SAVE_INTERMEDIATES = False # True = Guarda los 6 videos individuales. False = Solo la cuadrícula.
# ==============================================================================

def main():
    # 1. Validar origen de datos
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"No se encontró el video de entrada en: {VIDEO_PATH}")

    # 2. Configurar dispositivo (GPU CUDA si está disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Ejecutando inferencia en el dispositivo: {device.upper()}")

    # 3. Inicializar los 6 modelos YOLO
    print("[*] Cargando modelos YOLO...")
    models = {}
    for name, path in MODEL_CONFIGS.items():
        # Si el modelo personalizado no existe localmente, Ultralytics descargará los nativos automáticamente
        if "best" in path and not os.path.exists(path):
            print(f"[!] Advertencia: No se encontró {path}, usando weights por defecto para pruebas.")
            models[name] = YOLO("yolov26n.pt").to(device)
        else:
            models[name] = YOLO(path).to(device)

    # 4. Abrir video y capturar propiedades
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[*] Video original: {orig_width}x{orig_height} | {fps} FPS | {total_frames} frames totales.")

    # 5. Calcular resoluciones para la cuadrícula Full HD (1920x1080)
    # Una cuadrícula de 2x3 significa 3 columnas y 2 filas.
    # Cada celda medirá exactamente: 1920 / 3 = 640 de ancho, 1080 / 2 = 540 de alto.
    cell_width = 640
    cell_height = 540
    print(f"[*] Cada modelo se redimensionará individualmente a: {cell_width}x{cell_height}")

    # 6. Configurar VideoWriters temporales o individuales
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    intermediate_writers = {}
    temp_video_paths = {}

    for name in MODEL_CONFIGS.keys():
        # Si SAVE_INTERMEDIATES es False, igual creamos archivos temporales para armar el grid después
        suffix = "_output.mp4" if SAVE_INTERMEDIATES else f"_temp_{name}.mp4"
        temp_video_paths[name] = suffix
        intermediate_writers[name] = cv2.VideoWriter(suffix, fourcc, fps, (cell_width, cell_height))

    # 7. Bucle de procesamiento de frames (Un solo paso por el video)
    print("[*] Iniciando procesamiento e inferencia síncrona...")
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Procesando Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar el Frame Stride si es necesario
            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # Correr inferencia en paralelo para este frame exacto en los 6 modelos
            for name, model in models.items():
                # stream=True maneja mejor la memoria interna de la GPU
                results = model(frame, verbose=False, device=device)
                
                # Obtener el frame con las cajas ya dibujadas por Ultralytics
                annotated_frame = results[0].plot()

                # Redimensionar manteniendo la consistencia de la celda de la cuadrícula
                resized_frame = cv2.resize(annotated_frame, (cell_width, cell_height))

                # Añadir Overlay de texto (Fondo negro detrás del texto para alto contraste)
                text = f"Modelo: {name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Dibujar rectángulo de fondo para el texto
                cv2.rectangle(resized_frame, (10, 15), (20 + text_size[0], 25 + text_size[1]), (0, 0, 0), -1)
                # Escribir el nombre del modelo en texto verde/blanco brillante
                cv2.putText(resized_frame, text, (15, 20 + text_size[1]), font, font_scale, (0, 255, 0), thickness)

                # Guardar frame en el writer correspondiente
                intermediate_writers[name].write(resized_frame)

            frame_idx += 1
            pbar.update(1)

    # Liberar recursos de OpenCV
    cap.release()
    for writer in intermediate_writers.values():
        writer.release()
    cv2.destroyAllWindows()
    print("[*] Inferencia terminada. Videos individuales generados.")

    # 8. Combinar videos en una cuadrícula 2x3 usando FFmpeg (Complejidad de Tiempo Síncrona)
    print("[*] Construyendo cuadrícula 2x3 con FFmpeg...")
    
    # Lista ordenada de las rutas de videos individuales según el diccionario
    ordered_paths = [temp_video_paths[name] for name in MODEL_CONFIGS.keys()]
    
    try:
        # Cargamos los 6 inputs en FFmpeg
        input_videos = [ffmpeg.input(p) for p in ordered_paths]
        
        # Filtro xstack: junta los videos en posiciones específicas (X, Y)
        # Fila 1: (0,0), (640,0), (1280,0)
        # Fila 2: (0,540), (640,540), (1280,540)
        layout = "0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0"
        
        joined = ffmpeg.filter(input_videos, 'xstack', inputs=6, layout=layout)
        
        # Guardar el output final forzando codecs estándar
        process = (
            ffmpeg.output(joined, OUTPUT_GRID_PATH, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[+] ¡Éxito! Cuadrícula de comparación guardada en: {OUTPUT_GRID_PATH}")
        
    except ffmpeg.Error as e:
        print(f"[-] Error crítico de FFmpeg: {e.stderr.decode('utf8')}")
    
    finally:
        # 9. Limpieza opcional de archivos temporales intermediarios
        if not SAVE_INTERMEDIATES:
            print("[*] Limpiando archivos intermedios temporales...")
            for p in ordered_paths:
                if os.path.exists(p):
                    os.remove(p)
            print("[*] Limpieza completada.")

if __name__ == "__main__":
    main()