import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from scipy import stats
import scikit_posthocs as sp # Importa la biblioteca de post-hoc
import re

############# PUNTO 1

#Cargamos los datos como un dataframe de Pandas
file = 'DATOS_P1.tsv'
df = pd.read_csv(file,sep=',')

#Definimos las regiones
region_col = 'Geographic region'
rate_col = 'Attack rate (case/100 person)'

# Eliminamos filas con valores faltantes en las columnas clave (por si acaso)
df_clean = df.dropna(subset=[region_col, rate_col]).copy()

# La prueba Kruskal-Wallis requiere que los datos se separen en una lista por cada grupo.
grouped_data_lists = df_clean.groupby(region_col)[rate_col].apply(list).tolist()
groups_compared = df_clean[region_col].unique().tolist()

#Realización de la prueba y obtención del valor p
h_statistic, p_kruskal = stats.kruskal(*grouped_data_lists)

# 4. Ejecutar la prueba Post-Hoc de Dunn (con corrección de Bonferroni)
# La Prueba de Dunn requiere que los datos estén en formato de columna
# La función posthoc_dunn devuelve una matriz de valores p
dunn_results = sp.posthoc_dunn(
    a=df_clean,
    val_col=rate_col,
    group_col=region_col,
    p_adjust='bonferroni'  # Corrección estándar para múltiples comparaciones
)

# 5. Imprimir los resultados
print("\n" + "=" * 80)
print("Resultados de la Prueba Post-Hoc de Dunn (con corrección de Bonferroni)")
print("=" * 80)
print(f"Resultado de Kruskal-Wallis: H={h_statistic:.4f}, p={p_kruskal:.4f}")

print("\nMatriz de Valores p (Prueba de Dunn - Bonferroni):")
dunn_results.columns = groups_compared
dunn_results.index = groups_compared
print(dunn_results.round(4))
print("-" * 80)

# 6. Interpretación de la matriz
alpha = 0.05
print(f"Nivel de significancia (alpha): {alpha}")

print("\nInterpretación de las comparaciones:")
# Iterar sobre la matriz para identificar pares significativos
is_significant = (dunn_results < alpha)
found_significant = False

for i in range(len(groups_compared)):
    for j in range(i + 1, len(groups_compared)):
        region1 = groups_compared[i]
        region2 = groups_compared[j]
        p_val = dunn_results.iloc[i, j]

        if p_val < alpha:
            print(f"✅ {region1} vs. {region2}: p={p_val:.4f} (Diferencia significativa)")
            found_significant = True
        else:
            print(f"❌ {region1} vs. {region2}: p={p_val:.4f} (Sin diferencia significativa)")

if not found_significant:
    print("\nConclusión: No se encontró ninguna diferencia significativa entre pares de regiones.")


######## Punto 2

# 1. Definimos de la función de Von Bertalanffy (VBGF).
def von_bertalanffy(t, L_inf, k, t0):
    return L_inf * (1 - np.exp(-k * (t - t0)))

# Seleccionamos el archivo con los datos obtenidos del artículo.
file_name = 'DATOS_PUNTO_2.tsv'
title = "Ajuste de Crecimiento Von Bertalanffy"

# Cargamos el archivo .tsv como un dataframe de Pandas.
df = pd.read_csv(file_name, sep='\t')

# Seleccionamos los nombres de las columnas.
edad_col = 'Edad (años)'
longitud_col = 'Longitud (cm)'


# Asignamos y definimos las variables: x (Edad) e y (Longitud)
x = df[edad_col]
y = df[longitud_col]


# 3. Ajustamos la curva a el set de datos utilizando la función curve_fit
#   y la función del modelo de VBGF previamente definida.

# Valores iniciales (p0): [L_inf, k, t0], estos son los parámetros que definen la curva
#  de ajuste, y son los obtenidos por distintos métodos en el artículo.
L_inf=12.383
k=0.134
t0= -3.668
p0_vbgf = [12.383, 0.134, -3.668]

param, param_cov = curve_fit(von_bertalanffy, x, y, p0=p0_vbgf)


# Generar la curva ajustada
x_fit = np.linspace(x.min(), x.max(), 500)  # Para la gráfica del set de datos.
vbgf_curve = von_bertalanffy(x_fit, L_inf, k, t0)  # Para la grráfica de la curva de ajuste.

# 4. Visualización y Resultados

# Calculamos el Error Cuadrático Residual (RSS).
res = (y - von_bertalanffy(x, L_inf, k, t0))
rss = round(sum(res**2), 3)

# Creamos el gráfico de los puntos y la curva ajustada.
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, label='Datos Observados', color='black')
sns.lineplot(x=x_fit, y=vbgf_curve, color='blue', linewidth=2, label='Modelo VBGF Ajustado')

# Título y parámetros de las gráficas.
plt.title(f"{title}\n$L_\\infty$: {L_inf:.2f} cm, $k$: {k:.3f}, $t_0$: {t0:.2f} años "
          f"(RSS: {rss})", fontsize=14)
plt.xlabel("Edad (años)")
plt.ylabel("Longitud (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
sns.despine(offset=10, trim=True)
plt.legend()
plt.show()

# Impresión de  los resultados numéricos
print("\n Parámetros de Crecimiento VBGF:")
print(f"L_inf (Longitud Asintótica): {L_inf:.2f} cm")
print(f"k (Tasa de Crecimiento): {k:.4f}")
print(f"t0 (Edad a Tamaño Cero): {t0:.3f} años")
print(f"Error Cuadrático Residual (RSS): {rss}")


########### PUNTO 3

# 1. Definimos de las ecuaciones SIR
def sir_model(Y, t, beta, gamma):
    """
    Sistema de EDOs para el Modelo SIR.
    Y es un arreglo [S, I, R].
    """
    S, I, R = Y

    # Ecuaciones diferenciales:
    dSdt = -beta * (S/N) * I
    dIdt = beta * (S/N) * I - gamma * I
    dRdt = gamma * I

    return np.array([dSdt, dIdt, dRdt])


# 2. Parámetros y Condiciones Iniciales para el sistema de ecuaciones
# Parámetros del modelo (variables)
beta = 2  # Tasa de infección (por persona * día)
gamma = 0.85  # Tasa de recuperación (por día)
N = 50  # Población total (la suma de S + I + R debe ser constante)

# Condiciones iniciales
I0 = 1  # Inicialmente 1 infectados
R0 = 0  # Inicialmente 0 recuperados
S0 = N - I0 - R0  # El resto son susceptibles (49)
Y0 = [S0, I0, R0]  # Es el vector de condiciones iniciales [S0, I0, R0]

# Intervalo de tiempo definido para la simulación
tmax = 15. #Tiempo máximo que se observa en la simulación
Nt = 1000  #Número de puntos que se van a utilizar para resolver el sistema
t = np.linspace(0., tmax, Nt)

#3. Solución de las Ecuaciones
# Usamos odeint para resolver el sistema de EDOs
solution = integrate.odeint(sir_model, Y0, t,
                            args=(beta, gamma))

# Extraer los resultados para cada compartimento
S, I, R = solution.T

# 4. Visualización de los Resultados
plt.figure(figsize=(10, 6))

sns.lineplot(x=t, y=S, label='Susceptibles (S)', color='blue')
sns.lineplot(x=t, y=I, label='Infectados (I)', color='red')
sns.lineplot(x=t, y=R, label='Recuperados (R)', color='green')

plt.title('Modelo SIR de Dinámica de una Epidemia')
plt.xlabel('Tiempo (días)')
plt.ylabel('Número de Individuos')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()


