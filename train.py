import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# 1. Inicializar el tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# 2. Crear la función para predecir el sentimiento
def predict_sentiment(comment, model):
    # Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenizar el comentario
    encoding = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Mover los tensores y el modelo al dispositivo correspondiente (CPU o GPU)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    model = model.to(device)
    
    # Realizar la predicción sin necesidad de gradiente (para ahorrar memoria)
    with torch.no_grad():
        output = model(**encoding)
    
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]  # Obtener probabilidades
    return probabilities

# 3. Crear la clase Product
class Product:
    def __init__(self, name, comments, model):
        self.name = name
        self.comments = [comment for comment in comments if pd.notna(comment)]  # Filtrar comentarios no nulos
        self.average_rating = self.calculate_average_rating(model)  # Calcular el promedio de ratings

    def calculate_average_rating(self, model):
        # Promedio de ratings basado en la clasificación de los comentarios
        sentiment_scores = []
        for comment in self.comments:
            probabilities = predict_sentiment(comment, model)
            sentiment_scores.append(probabilities[1])  # Tomar la probabilidad de ser "Positive"
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# 4. Cargar el dataset de productos
products_df = pd.read_csv('products.csv')  # Asegúrate de que el archivo esté en el mismo directorio
products = []

# 5. Cargar el dataset de comentarios de Amazon
data = pd.read_csv('amazon_cells_labelled.csv', header=None, names=['comment', 'label'])

# 6. Eliminar filas con valores nulos (si existen)
data.dropna(inplace=True)

# 7. Dividir el dataset en conjunto de entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# 8. Tokenización
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 9. Crear un dataset compatible con PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# 10 Cargar el modelo
model = RobertaForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment', 
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 11 Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=500,                 
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluar el modelo cada epoch
)

# 12 Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 13 Entrenar el modelo
trainer.train()

# 14 Guardar el modelo y el tokenizador explícitamente en formato PyTorch
model.save_pretrained('./results', safe_serialization=False)
tokenizer.save_pretrained('./results')

# Verificar si el archivo 'pytorch_model.bin' se guardó correctamente
import os
if not os.path.isfile('./results/pytorch_model.bin'):
    raise ValueError("Error: No se guardó el archivo 'pytorch_model.bin'. Verifica si el modelo se guardó correctamente.")
else:
    print(f"Modelo guardado exitosamente en ./results/pytorch_model.bin")

# 15. Mostrar resultados ordenados
if __name__ == "__main__":
    # Ejemplo para evaluar un comentario
    comment = "this is very good"
    probabilities = predict_sentiment(comment, model)
    print(f"Probabilidades para el comentario '{comment}': {probabilities}")  # [prob_negativa, prob_positiva]

    # Ejemplo pequeño para evaluar si un comentario es positivo o negativo
    test_comment = "I love this product, it's amazing!"
    test_probabilities = predict_sentiment(test_comment, model)
    positive_probability = test_probabilities[1]

    if positive_probability > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print(f"Comentario: '{test_comment}'")
    print(f"Probabilidad de ser positivo: {positive_probability:.4f}")
    print(f"Clasificación: {sentiment}")

    # Ordenar productos por rating promedio
    for index, row in products_df.iterrows():
        comments = [row.get('comment1'), row.get('comment2'), row.get('comment3'), row.get('comment4'), row.get('comment5')]
        product = Product(row['product_name'], comments, model)
        products.append(product)

    sorted_products = sorted(products, key=lambda p: p.average_rating, reverse=True)

    print("\nProductos ordenados por calificación promedio de sentimientos positivos:")
    for product in sorted_products:
        print(f"{product.name}: {product.average_rating:.4f}")