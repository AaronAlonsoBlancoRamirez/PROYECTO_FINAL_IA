import os
import random
from flask import Flask, request, render_template
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo y el tokenizer desde la carpeta 'results'
model_directory = './results'

# Cargar el tokenizador y el modelo desde el directorio de resultados
print("Cargando el tokenizador y el modelo...")
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_directory)
    model = RobertaForSequenceClassification.from_pretrained(model_directory)
    print("Modelo y tokenizador cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o el tokenizador: {e}")

# Verificar si hay GPU disponible y mover el modelo al dispositivo adecuado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Usando el dispositivo: {device}")

# Función para predecir el sentimiento
def predict_sentiment(comment):
    """Predicción del sentimiento de un comentario."""
    # Tokenizar el comentario
    print(f"Tokenizando el comentario: '{comment}'")
    encoding = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Mover los tensores al dispositivo adecuado (CPU o GPU)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    model.to(device)  # Asegurarse de que el modelo también está en el dispositivo adecuado
    
    # Realizar la predicción sin necesidad de gradiente
    with torch.no_grad():
        output = model(**encoding)
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    # Depuración: Imprimir los logits y las probabilidades
    print(f"Logits: {logits}")
    print(f"Probabilidades: {probabilities}")
    
    # Regresar las probabilidades para cada clase
    return probabilities

# Ruta principal de la aplicación Flask
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        comment = request.form["comment"]
        probabilities = predict_sentiment(comment)
        labels = ['Negative', 'Positive']

        # Determinar si el comentario es positivo o negativo según las probabilidades
        sentiment = 'Positive' if probabilities[1] > probabilities[0] else 'Negative'

        # Agregar las probabilidades de ambas clases al resultado
        result = {
            'label': sentiment,
            'prob_negativo': f"{probabilities[0]:.4f}",
            'prob_positivo': f"{probabilities[1]:.4f}"
        }
        
        # Depuración: Imprimir el resultado antes de pasarlo al HTML
        print("Resultado del análisis:")
        print(result)
    
    return render_template("index.html", result=result)

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)
