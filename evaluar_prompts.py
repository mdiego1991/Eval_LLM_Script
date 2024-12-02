import pandas as pd
import requests
import json
import os

# Configuración de la API de Ollama
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Nombre del archivo CSV que contiene los prompts y atributos
PROMPTS_FILE = "prompts_evaluacion_modelos_ia.csv"

# Verificar si el archivo CSV existe
if not os.path.exists(PROMPTS_FILE):
    print(f"Error: No se encontró el archivo '{PROMPTS_FILE}'. Asegúrate de crearlo en el mismo directorio que este script.")
    exit(1)

# Leer los prompts desde el archivo CSV
try:
    prompts_data = pd.read_csv(PROMPTS_FILE)
except Exception as e:
    print(f"Error al leer el archivo '{PROMPTS_FILE}': {e}")
    exit(1)

if prompts_data.empty:
    print(f"Error: El archivo '{PROMPTS_FILE}' está vacío o no contiene datos válidos.")
    exit(1)

# Verificar que el archivo contenga las columnas necesarias
required_columns = {"prompt", "atributo", "resultado"}
if not required_columns.issubset(prompts_data.columns):
    print(f"Error: El archivo '{PROMPTS_FILE}' debe contener las columnas: {', '.join(required_columns)}.")
    exit(1)

# Función para generar respuestas usando la API de Ollama
def generate_response(prompt, model=MODEL_NAME):
    try:
        # Validar que el prompt no contenga valores no JSON-compliant
        data = {
            "model": model,
            "prompt": prompt
        }

        # Asegurarse de que no hay valores NaN o fuera de rango
        if not isinstance(data["prompt"], str) or data["prompt"].strip() == "":
            return "Error: El prompt está vacío o no es válido."
        
        # Enviar la solicitud
        response = requests.post(API_URL, json=data)
        
        if response.status_code == 200:
            try:
                # Procesar flujo de respuesta JSON concatenado
                raw_responses = response.text.splitlines()
                reconstructed_text = ""
                for raw in raw_responses:
                    try:
                        partial = json.loads(raw)
                        if "response" in partial:
                            reconstructed_text += partial["response"]
                        if partial.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                return reconstructed_text
            except Exception as e:
                return f"Error al procesar la respuesta: {str(e)}"
        else:
            return f"Error: {response.status_code}, {response.text}"
    except ValueError as ve:
        return f"Error en los datos: {str(ve)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

# Crear una lista para almacenar los resultados
results = []

# Evaluar los prompts
for idx, row in prompts_data.iterrows():
    prompt = row["prompt"]
    quality_attribute = row["atributo"]
    expected_result = row["resultado"]
    print(f"Evaluando prompt {idx + 1}/{len(prompts_data)}: {prompt} ({quality_attribute})")
    response = generate_response(prompt)
    results.append({
        "Id": idx + 1,
        "Prompt": prompt,
        "Atributo": quality_attribute,
        "Resultado Esperado": expected_result,
        "Respuesta": response
    })

# Guardar los resultados en un archivo CSV para análisis posterior
df_results = pd.DataFrame(results)
df_results.to_csv("resultados_prompts.csv", index=False, encoding="utf-8-sig")

print("Evaluación completada. Resultados guardados en 'resultados_prompts.csv'.")
