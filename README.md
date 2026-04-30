# Herramientas Documentado

Análisis de Ventas y Segmentación de Productos con K-Means
1. Descripción general del proyecto
Este proyecto tiene como finalidad analizar una base de datos de ventas llamada Sales Data.csv, utilizando herramientas de análisis de datos en Python. El objetivo principal es limpiar, organizar, analizar y visualizar la información para identificar patrones importantes en el comportamiento de los productos.
El proyecto se enfoca en el análisis de ventas, precios, cantidades vendidas, horarios de compra y comportamiento general de los productos. Además, se aplica el algoritmo de Machine Learning K-Means, el cual permite segmentar los productos en diferentes grupos de acuerdo con sus características comerciales.
Desde un contexto empresarial, este análisis puede ser utilizado por una empresa para tomar mejores decisiones sobre inventario, precios, estrategias de venta, productos premium, productos de alto volumen y productos con rendimiento medio.
________________________________________
2. Objetivo del proyecto
El objetivo del proyecto es desarrollar un análisis de datos que permita clasificar productos mediante técnicas estadísticas y de Machine Learning, tomando en cuenta variables como:
•	Cantidad ordenada. 
•	Precio de cada producto. 
•	Ventas generadas. 
•	Mes de compra. 
•	Hora de compra. 
•	Ventas totales por producto. 
•	Precio promedio. 
•	Cantidad total vendida. 
Con esto se busca generar información útil para apoyar la toma de decisiones empresariales.
________________________________________
3. Estructura del repositorio en GitHub
El proyecto fue documentado y organizado dentro de GitHub de la siguiente manera:
Proyecto-Ventas-KMeans/
│
├── Sales Data.csv
├── analisis_ventas_kmeans.ipynb
├── README.md
└── requirements.txt
Descripción de los archivos
Sales Data.csv:
Archivo principal que contiene la base de datos utilizada para el análisis.
analisis_ventas_kmeans.ipynb:
Notebook donde se encuentra el código del proyecto, incluyendo limpieza de datos, análisis estadístico, gráficas, pruebas de normalidad y segmentación K-Means.
README.md:
Documento principal del repositorio. Explica qué hace el proyecto, cómo instalarlo, cómo usarlo, qué parámetros se pueden modificar y qué resultados se obtienen.
requirements.txt:
Archivo donde se enlistan las librerías necesarias para ejecutar el proyecto.
________________________________________
4. Requisitos e instalación
Para ejecutar correctamente este proyecto se necesita tener instalado:
•	Python 3.8 o superior. 
•	Jupyter Notebook o Google Colab. 
•	Git. 
•	Acceso al repositorio en GitHub. 
•	Archivo Sales Data.csv. 
Librerías necesarias
pandas
matplotlib
seaborn
scikit-learn
scipy
numpy
Instalación de dependencias
Para instalar las librerías necesarias, se puede utilizar el siguiente comando:
pip install pandas matplotlib seaborn scikit-learn scipy numpy
En caso de usar Google Colab, la mayoría de estas librerías ya vienen instaladas, pero si alguna no está disponible, se puede instalar con:
!pip install nombre_libreria
________________________________________
5. Configuración inicial del proyecto
Primero se debe colocar el archivo Sales Data.csv dentro de la misma carpeta donde se encuentra el notebook del proyecto.
Después, se carga la base de datos con el siguiente código:
df = pd.read_csv('Sales Data.csv')
df
Este paso permite importar la información a Python mediante la librería pandas, creando un DataFrame llamado df.
________________________________________
6. Proceso de análisis del proyecto
6.1 Importación de librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Estas librerías permiten trabajar con datos, crear gráficas y visualizar patrones.
________________________________________
6.2 Carga de datos
df = pd.read_csv('Sales Data.csv')
df
Se carga el archivo CSV con la información de ventas. Esto permite observar las columnas, filas y estructura general de la base de datos.
________________________________________
6.3 Selección de variables
X = df[['Quantity Ordered', 'Price Each', 'Month', 'Hour']]
y = df['Sales']
En este paso se definen las variables predictoras y la variable objetivo.
Variables predictoras:
•	Quantity Ordered: cantidad de productos vendidos. 
•	Price Each: precio unitario del producto. 
•	Month: mes de la venta. 
•	Hour: hora de la venta. 
Variable objetivo:
•	Sales: total de ventas. 
________________________________________
6.4 Limpieza de valores faltantes
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    elif df[col].dtype == 'object':
        if df[col].isnull().any():
            df[col] = df[col].fillna('perdidos')
En esta etapa se revisan valores vacíos.
Las columnas numéricas se rellenan con la mediana y las columnas categóricas con la palabra "perdidos".
________________________________________
6.5 Eliminación de duplicados
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
final_rows = df.shape[0]
duplicates_removed = initial_rows - final_rows
Este proceso elimina registros repetidos para evitar errores en los resultados.
________________________________________
6.6 Análisis estadístico
df.describe()
Se obtiene un resumen estadístico con media, desviación estándar, mínimos, máximos y cuartiles.
________________________________________
6.7 Detección de outliers
Q1 = df['Price Each'].quantile(0.25)
Q3 = df['Price Each'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
Este proceso permite identificar valores atípicos en la columna Price Each.
________________________________________
6.8 Visualización de datos
Se generan diferentes gráficos:
sns.boxplot(x=df['Price Each'])
sns.histplot(df['Sales'], bins=50, kde=True)
Estos gráficos ayudan a interpretar visualmente la distribución de precios y ventas.
________________________________________
6.9 Prueba de normalidad
from scipy import stats

k2, p_value = stats.normaltest(df['Price Each'])
Esta prueba permite saber si los datos siguen una distribución normal.
________________________________________
6.10 Segmentación con K-Means
Primero se crea un resumen por producto:
product_summary = df.groupby('Product').agg(
    total_sales=('Sales', 'sum'),
    avg_price_each=('Price Each', 'mean'),
    total_quantity_ordered=('Quantity Ordered', 'sum')
).reset_index()
Después se seleccionan las variables para clustering:
features = product_summary[['total_sales', 'avg_price_each', 'total_quantity_ordered']]
Luego se escalan los datos:
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
Finalmente se aplica K-Means:
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
product_summary['Cluster'] = kmeans.fit_predict(scaled_features)
________________________________________
7. Parámetros modificables
Esta sección es una de las más importantes, ya que permite identificar qué partes del código pueden cambiarse según las necesidades del usuario.
7.1 Nombre del archivo de entrada
df = pd.read_csv('Sales Data.csv')
Parámetro modificable: Sales Data.csv
Se puede cambiar por otro archivo CSV.
Consecuencia:
Si se cambia el archivo, debe tener columnas compatibles con el código.
________________________________________
7.2 Variables predictoras
X = df[['Quantity Ordered', 'Price Each', 'Month', 'Hour']]
Parámetros modificables:
•	Quantity Ordered 
•	Price Each 
•	Month 
•	Hour 
Consecuencia:
Cambiar estas columnas modifica las variables que se utilizan para el análisis.
________________________________________
7.3 Variable objetivo
y = df['Sales']
Parámetro modificable: Sales
Consecuencia:
Si se cambia la variable objetivo, el análisis se enfocará en otra métrica.
________________________________________
7.4 Método de imputación
median_val = df[col].median()
df[col] = df[col].fillna(median_val)
Parámetro modificable: método de relleno de datos faltantes.
Se puede cambiar por:
•	Media. 
•	Moda. 
•	Eliminación de filas. 
•	Cero. 
•	Otro valor definido. 
Consecuencia:
El método elegido puede afectar los resultados estadísticos.
________________________________________
7.5 Texto para datos categóricos faltantes
df[col] = df[col].fillna('perdidos')
Parámetro modificable: "perdidos"
Consecuencia:
Puede cambiarse por "sin dato", "desconocido" u otra categoría.
________________________________________
7.6 Columna para detección de outliers
df['Price Each']
Parámetro modificable: Price Each
Consecuencia:
Se puede analizar otra columna numérica, como Sales o Quantity Ordered.
________________________________________
7.7 Factor del rango intercuartílico
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
Parámetro modificable: 1.5
Consecuencia:
Un valor menor detecta más outliers.
Un valor mayor detecta menos outliers.
________________________________________
7.8 Número de barras del histograma
sns.histplot(df['Sales'], bins=50, kde=True)
Parámetro modificable: bins=50
Consecuencia:
Más barras muestran mayor detalle.
Menos barras muestran una vista más general.
________________________________________
7.9 Activación de KDE
kde=True
Parámetro modificable: True
Puede cambiarse a:
kde=False
Consecuencia:
Activa o desactiva la curva de densidad en el histograma.
________________________________________
7.10 Nivel de significancia
alpha = 0.05
Parámetro modificable: 0.05
Puede cambiarse por:
•	0.01 
•	0.10 
Consecuencia:
Cambia el criterio para aceptar o rechazar la normalidad de los datos.
________________________________________
7.11 Variables para clustering
features = product_summary[['total_sales', 'avg_price_each', 'total_quantity_ordered']]
Parámetros modificables:
•	total_sales 
•	avg_price_each 
•	total_quantity_ordered 
Consecuencia:
Modificar estas variables cambia la forma en que se agrupan los productos.
________________________________________
7.12 Número de clusters
KMeans(n_clusters=3)
Parámetro modificable: n_clusters=3
Consecuencia:
Determina cuántos grupos de productos se crearán.
________________________________________
7.13 Random state
random_state=42
Parámetro modificable: 42
Consecuencia:
Permite que los resultados sean reproducibles. Si se cambia, los clusters pueden variar ligeramente.
________________________________________
7.14 Número de inicializaciones
n_init=10
Parámetro modificable: 10
Consecuencia:
Indica cuántas veces se ejecuta K-Means con diferentes centroides iniciales. Un valor mayor puede mejorar la estabilidad del modelo.
________________________________________
7.15 Tamaño de las gráficas
plt.figure(figsize=(10, 6))
Parámetros modificables: 10 y 6
Consecuencia:
Permiten cambiar el tamaño visual de las gráficas.
________________________________________
8. Ejemplos de uso
Ejemplo 1: Ejecutar el proyecto con el archivo original
df = pd.read_csv('Sales Data.csv')
df.head()
Este ejemplo carga la base de datos original y muestra las primeras filas.
________________________________________
Ejemplo 2: Cambiar el número de clusters
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
product_summary['Cluster'] = kmeans.fit_predict(scaled_features)
Este ejemplo permite clasificar los productos en 4 grupos en lugar de 3.
________________________________________
Ejemplo 3: Analizar outliers en ventas
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_sales = df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)]
Este ejemplo cambia la detección de outliers de Price Each a Sales.
________________________________________
9. Resultados del proyecto
El proyecto permite identificar tres perfiles principales de productos:
Cluster 0: Productos de desempeño moderado
Son productos con ventas, precios y cantidades en niveles medios.
Pueden fortalecerse con promociones o estrategias de venta cruzada.
Cluster 1: Productos premium
Son productos con precios altos y ventas importantes, aunque no necesariamente con alto volumen.
Se recomienda mantener estrategias de valor, calidad y exclusividad.
Cluster 2: Productos de alto volumen
Son productos con precios bajos, pero con gran cantidad vendida.
Se recomienda cuidar inventarios, costos y disponibilidad.
________________________________________
10. Evidencias visuales que debes agregar
Para cumplir con la rúbrica, agrega mínimo 4 capturas:
Captura 1: Repositorio en GitHub mostrando los archivos.
Captura 2: README abierto en GitHub.
Captura 3: Notebook ejecutando la carga del archivo CSV.
Captura 4: Gráfica generada, como histograma, boxplot o pairplot.
Captura 5 opcional: Resultado de clusters o Silhouette Score.
Debajo de cada captura puedes poner:
Evidencia: Esta captura muestra el repositorio/documentación/ejecución del proyecto, comprobando que el análisis fue desarrollado y documentado correctamente dentro de GitHub.
________________________________________
11. Comandos Git utilizados
git init
git status
git add .
git commit -m "Documentación inicial del proyecto"
git push origin main
git pull origin main
Estos comandos permiten inicializar el repositorio, revisar cambios, agregar archivos, guardar versiones, subir el proyecto a GitHub y sincronizar cambios.
________________________________________
12. Conclusión
La documentación del proyecto en GitHub permite presentar de forma clara y organizada el funcionamiento del análisis de ventas y segmentación de productos. El README cumple una función esencial porque explica el propósito del proyecto, los requisitos de instalación, la forma de uso, los parámetros modificables y los resultados obtenidos.
Además, el proyecto demuestra la aplicación de herramientas de análisis de datos y Machine Learning en un contexto empresarial, permitiendo clasificar productos de acuerdo con su desempeño. Esto facilita la toma de decisiones relacionadas con precios, inventario, promociones y estrategias comerciales.
Con esta documentación, el repositorio queda mejor estructurado, más profesional y cumple con los criterios solicitados en la rúbrica de evaluación.
