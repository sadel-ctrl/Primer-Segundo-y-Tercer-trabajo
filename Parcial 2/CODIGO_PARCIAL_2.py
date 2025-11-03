from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


######################### PARTE 1 ################################
# 1. Cargar el archivo TSV
# Nota: La ruta debe apuntar a tu archivo.
file_path = 'clinical.tsv'
try:
    df_completo = pd.read_csv(file_path, sep='\t')
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {file_path}")
    exit()

# 2. Tomar solo las primeras 50 filas
n_filas = 50
df_50_primeras = df_completo.head(n_filas)

print(f"Número total de filas en el archivo original: {len(df_completo)}")
print(f"Número de filas en el nuevo DataFrame: {len(df_50_primeras)}")

# Imprime las 50 filas con formato Markdown para una mejor legibilidad en la consola
print("\n--- Primeras 50 Líneas ---")
print(df_50_primeras)

###### Figura 1
g1 = sns.countplot(x='vital_status.demographic',
                   hue='figo_stage.diagnoses',
                   data=df_50_primeras,
                   palette='hls'
                   )
plt.title("Actual vital status and its relation with the stage of diagnosis",
          color='seagreen',
          size=15)
plt.xlabel("Actual vital status")
plt.ylabel("Count")
g1.legend(title="Stage of diagnosis")
plt.grid(axis='y',linestyle='--', color='black')
plt.show()



####### Figura 2

sns.set_style("whitegrid")

g2 = sns.boxplot(data=df_50_primeras,
            y='ethnicity.demographic',
            x='days_to_last_follow_up.diagnoses',
            palette='plasma')

plt.title("Relation of the ethnicity of the patients with the days of their last follow up",
          size=10, weight='bold', ha='center')

plt.xlabel("Days passed from the last follow up", size=10, weight='bold')
plt.ylabel("Ethnicity", size=9, weight='bold')
plt.grid(axis="both",linestyle='-')
sns.set_theme(style="ticks")

etiquetas_completas = ['Hispanic \n or Latino',
                       'Not \n reported',
                       'Non-\nHispanic \n or Latino']
orden_etiquetas = etiquetas_completas # Para asegurar el orden


g2.set_yticklabels(orden_etiquetas, fontsize=8)
plt.tight_layout()
g2.set_facecolor('lemonchiffon')

sns.stripplot(data=df_50_primeras,
            y='ethnicity.demographic',
            x='days_to_last_follow_up.diagnoses',
            marker='*',
            color='springgreen',
              size=9
              )
plt.show()



##### Figura 3
# intermediate_dimension.sample y age_at_index.demographic

#Columnas para el Heatmap
COL_X = 'age_at_index.demographic'
COL_Y = 'intermediate_dimension.samples'

# 1. Cargar el archivo .tsv en un DataFrame
# Usamos sep='\t' porque es un archivo TSV
df = pd.read_csv('clinical.tsv', sep='\t', low_memory=False)

# 2. Crear el sub-DataFrame con las dos columnas
# Esto también elimina las filas con valores faltantes (NaN) en esas columnas
df_heatmap = df[[COL_Y, COL_X]].dropna()

# 3. Crear la Tabla de Contingencia (Matriz de Frecuencias)
# Esta matriz cuenta cuántos pacientes hay en cada combinación de Etapa y Etnicidad.
contingency_table = pd.crosstab(
    df_heatmap[COL_Y],
    df_heatmap[COL_X],
    margins=False # No incluye las filas/columnas de 'Total'
)

# 1. Definir la figura y los ejes
plt.figure(figsize=(10, 7))

# 2. Generar el Heatmap
sns.heatmap(
    contingency_table,
    annot=False,          # Muestra el número de conteos en cada celda
    fmt='d',             # Formato 'd' (entero) para los conteos
    cmap='viridis',     # Paleta de colores.
    linewidths=.5,       # Añade líneas divisorias
    linecolor='purple',   # Color de las líneas
    cbar_kws={'label': 'Patients count'} # Etiqueta de la barra de color
)

# 3. Añadir títulos y etiquetas
plt.title(
    f'Dimension per age frecuency',
    fontproperties='serif',
    fontsize=19,
    weight='bold', # Poner el título en negrilla
     )

plt.xlabel("Age", fontsize=12, weight='bold')
plt.ylabel("Intermediate diemension samples ", fontsize=14, weight='bold')

# 4. Ajustar el diseño para asegurar que todas las etiquetas se vean
plt.tight_layout()
plt.show()


###### Figura 4

#year_of_birth.demographic y year_of_death.demographic y name.tissue_source_site
# Columnas a usar
COL_X = 'year_of_birth.demographic'
COL_Y = 'year_of_death.demographic'
COL_HUE = 'name.tissue_source_site'

# Cargar el archivo .tsv
df = pd.read_csv('clinical.tsv', sep='\t', low_memory=False)

# Limpiar los datos: Convertir a numérico y eliminar valores no válidos (NaN)
# La columna de edad a veces viene como flotante, la convertimos a entero si es posible
df[COL_X] = pd.to_numeric(df[COL_X], errors='coerce')
df[COL_Y] = pd.to_numeric(df[COL_Y], errors='coerce')

# Crear el DataFrame limpio para el scatterplot (eliminando filas con valores faltantes)
df_scatter = df[[COL_X, COL_Y, COL_HUE]].dropna()

# Opcional: Filtramos la etnicidad 'not reported' para mejorar la visualización del color
df_scatter = df_scatter[df_scatter[COL_HUE] != 'not reported']

# 1. Definir la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Generar el Scatterplot
# 'data': el DataFrame limpio
# 'x' y 'y': las columnas numéricas
# 'hue': la columna categórica que determina el color
sns.scatterplot(
    data=df_scatter,
    x=COL_X,
    y=COL_Y,
    hue=COL_HUE,
    ax=ax,
    s=70,          # Tamaño del punto
    alpha=0.6,     # Transparencia del punto (útil para ver la densidad)
    palette='terrain' # Paleta de colores
)

# 3. Añadir títulos y etiquetas
fig.suptitle(
    'Relation of year of birth and year of death by tissue source site',
    fontsize=16,
    weight='bold',
    y=0.97
)
ax.set_title('')

# Etiquetas de los ejes
ax.set_xlabel("Year of birth", fontsize=12)
ax.set_ylabel("Year of death", fontsize=12)

# 4. Ajustar la leyenda
# Colocar la leyenda fuera del área del gráfico para no interferir con los puntos
ax.legend(title='Tissue source site', bbox_to_anchor=(1.05, 1),
          loc='upper left',)
ax.set_facecolor('thistle')

plt.tight_layout()
plt.show()



###### Figura 5

# Columnas a usar
COL_NUM = 'age_at_index.demographic' # Variable numérica: Edad
COL_CAT = 'ethnicity.demographic'    # Variable categórica: Etnicidad

# 1. Cargar el archivo .tsv
df = pd.read_csv('clinical.tsv', sep='\t', low_memory=False)

# 2. Limpiar y seleccionar datos
# Convertir la columna de edad a numérica (manejar posibles errores con 'coerce')
df[COL_NUM] = pd.to_numeric(df[COL_NUM], errors='coerce')

# Crear el DataFrame limpio: eliminar filas con valores faltantes (NaN)
# y la etnicidad 'not reported' para una mejor comparación.
df_displot = df[[COL_NUM, COL_CAT]].dropna()
df_displot = df_displot[df_displot[COL_CAT] != 'not reported']

# Creamos la Figura y los Ejes
plt.figure(figsize=(7,3))

# Generar el Displot
g5 = sns.displot(
    data=df_displot,
    x=COL_NUM,
    hue=COL_CAT,    # Divide la distribución por Etnicidad
    kind='hist',    # Tipo de gráfico: histograma
    kde=True,       # Añade la curva de densidad (KDE)
    bins=20,        # Número de barras del histograma
    palette='ocean',
    height=3.5,       # Altura de la figura
    aspect=1.5      # Relación de aspecto (ancho/alto)
)

plt.title("Relation between age and etchnicity",
          size=11,  weight='black', ha='center', y=0.98)

# Etiquetas de los ejes
g5.set_xlabels("Age(years)", fontsize=11,  fontproperties='sans')
g5.set_ylabels("Frecuency", fontsize=11, fontproperties='sans')

# 1. Definir las nuevas etiquetas que quieres mostrar en la leyenda
# (Puedes usar saltos de línea '\\n' aquí si lo deseas, o simplemente limpiarlas)
nuevas_etiquetas = [
    'Hispanic\nor Latino',
    'Non-Hispanic\nor Latino',
    'Not reported']
# 2. Acceder y modificar las etiquetas de la leyenda de g5
# Nota: La lista de etiquetas se invierte automáticamente a veces,
# por lo que podrías necesitar invertir 'nuevas_etiquetas' si no coinciden.
for t, l in zip(g5.legend.texts, nuevas_etiquetas):
    t.set_text(l)

g5.legend.set_title('Ethnicity')
g5.legend.get_title().set_fontweight('bold')

# 1. Accede al primer (y único) eje del FacetGrid
ax = g5.axes.flat[0]

# 2. Aplica el color de fondo solo al ÁREA DE TRAZADO (Axes)
ax.set_facecolor('pink')

plt.grid(axis='both',linestyle=':', color='dimgrey')

# Mostrar el gráfico
plt.show()


############################# PARTE 2 #################################

def GC_content(sequence):
    count_A = sequence.count('A')
    count_G = sequence.count('G')
    count_C = sequence.count('C')
    count_T = sequence.count('T')
    GC_content = (count_G + count_C) / len(sequence)
    GC_content = round(100 * GC_content, 2)
    return GC_content
    pass

filename ='../Secuencias/ecoli_seq.fasta'
sequence = str()
with open(filename) as file:
    next(file)
    for line in file:
        line = line.strip()
        # print(line)
        sequence += line

sequence = list(sequence)
random.shuffle(sequence)
sequence = "".join(sequence)[:100000]
gc_list = []
seqlen = 100
for repetition in range(1000):
    random_seq = list(sequence)
    random.shuffle(random_seq)
    random_seq = "".join(random_seq)[:seqlen]
    gc_value = GC_content(random_seq)
    gc_list.append(gc_value)
    print(gc_value)
gc_value = pd.Series(gc_list)
p_value = sum(gc_value >= 60)/repetition

plt.figure(figsize=(10, 6), facecolor='lavender')
sns.histplot(gc_list, kde=True, bins=30, color='limegreen', edgecolor='black',)
plt.grid(axis='both',linestyle='--', color='black')

# Marcar el umbral usado para el cálculo del valor p
plt.axvline(60, color='red', linestyle='--', linewidth=2, label=f'Umbral GC% {60}%)')

# Añadir el valor p como texto en la gráfica
plt.text(0.95, 0.95,
         f'$P_{{empírico}}$ ($\geq$ {60}% GC) = {p_value:.4f}',
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

plt.title(f'Distribución de Contenido GC', weight='black')
plt.xlabel('Contenido GC (%)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(axis='both', alpha=0.5)
plt.show()

