# Análisis de sentimientos en Tweets de opiniones hacia Aerolíneas
Pequeño código para clasificar los sentimientos (positivos, neutrales, negativos) en un tweet. En este caso son tweets de conformidad con diversas aerolíneas. Se utiliza un proceso que limpia la información de los tweets usando `stopwords` y `n-gramas` antes de entrenar. El modelo propuesto aún no genera resultados tan precisos y se propone solamente como ejercicio de aprendizaje en el área de Aprendizaje de Máquina y Procesamiento de Lenguaje Natural.

### Requisitos:
- Python >=3.8.3
    - Instalar librerías requeridas en el archivo `requirements.txt`
- nltk_data corpus de stopwords
    - Se adjunto una carpeta (`nltk_data`) el cual puede ser utilizado en el path de búsqueda de la librería de Python **nltk** se debe agregar este path a la variable de entorno, ej. `NLTK_DATA=nltk_data`
- Dataset con los tweets a entrenar el modelo
    - Se adjunta una carpeta (`datasets`) con el ejemplo del csv a utilizar

### Ejecución
``` bash
python project/sentimental_analysis.py datasets/Tweets.csv (--debug)
```
_**El `(--debug)` al final es opcional y se usa para especificar que se impriman datos **DEBUG** en la ejecución del programa**_
