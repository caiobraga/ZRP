import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping, MultiPolygon
from shapely.ops import unary_union
from pathlib import Path

class ZooInstanceGenerator:
    def __init__(self, input_folder="zoos_images", output_folder="instances"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def process_zoo_image(self, image_path):
        image = cv2.imread(str(image_path))

        # Detecção de obstáculos
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        internal_contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        obstacle_polygons = []

        for cnt in internal_contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            poly = Polygon(cnt.squeeze()).simplify(1.0)
            if poly.is_valid and poly.area >= 50:
                obstacle_polygons.append(poly)
                obstacles.append(mapping(poly))  # <=== Agora cada obstáculo é um GeoJSON completo

        if not obstacle_polygons:
            print(f"Nenhum obstáculo encontrado em {image_path.name}")
            return

        # União dos obstáculos para formar o contêiner (zoo boundary)
        union = unary_union(obstacle_polygons)
        zoo_polygon = union.buffer(30).simplify(3.0)

        if isinstance(zoo_polygon, MultiPolygon):
            zoo_polygon = max(zoo_polygon.geoms, key=lambda p: p.area)

        # --- PLOT VISUALIZAÇÃO ---
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Imagem Original")

        axs[1].imshow(clean, cmap='gray')
        axs[1].set_title("Obstáculos Detectados")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_mask1 = np.array([85, 30, 180])   # azul claro
        upper_mask1 = np.array([130, 255, 255])
        lower_mask2 = np.array([25, 30, 150])   # verde/amarelo
        upper_mask2 = np.array([80, 255, 255])
        mask1 = cv2.inRange(hsv, lower_mask1, upper_mask1)
        mask2 = cv2.inRange(hsv, lower_mask2, upper_mask2)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        axs[2].imshow(combined_mask, cmap='gray')
        axs[2].set_title("Máscara Colorida (Comparação)")

        axs[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[3].set_title("Contornos Finais")

        if zoo_polygon.is_valid:
            x, y = zoo_polygon.exterior.xy
            axs[3].plot(x, y, color='blue', linewidth=2)
        for obs in obstacle_polygons:
            if obs.is_valid:
                x, y = obs.exterior.xy
                axs[3].plot(x, y, color='red', linewidth=1)

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # --- Salvar JSON no formato esperado pelo visualizador ---
        result = {
            "zoo_name": image_path.stem,  # Nome da instância baseado no nome da imagem
            "container": mapping(zoo_polygon),  # GeoJSON Polygon do zoo
            "obstacles": obstacles  # Lista de GeoJSON Polygons para obstáculos
        }

        output_path = self.output_folder / f"{image_path.stem}_instance.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Instância salva em: {output_path}")

    def generate_all_instances(self):
        for image_file in self.input_folder.glob("*.png"):
            self.process_zoo_image(image_file)

if __name__ == "__main__":
    generator = ZooInstanceGenerator(
        input_folder="zoos_images",
        output_folder="instances"
    )
    print("Iniciando geração de instâncias...")
    generator.generate_all_instances()
    print("\nGeração concluída.")
