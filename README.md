# Primer trabajo
Esta es la parte 1 del primer tarbajo del curso, aquí voy a mostrar uuna rutina para ejecutar en Pyhton, que permite transformar archivos de formato .gtf a formato .tsv.

La rutina mostrada fue generada con ayuda la IA, específicamente de Chatgpt y Gemini.

## Acerca del archivo

- El archivo que voy a utilizar hace parte del genoma de el platelminto japonés *flatSchistosoma japonicum*, que fue tomado de la página web de la UCSC Genome Browser.
- La idea al transformar este archivo a formato .tsv es poder visualizar la información de manera más organizada y en 11 columnas diferentes, separadas claramente, cada una conteniendo la siguiente información:

  - seqname: nombre de la secuencia de referencia.
  - source: base de datos donde fue generada.
  - feature: característica a la que se refiere.
  - start: posición inicial del cromosoma.
  - end: posición final del cromosoma.
  - score: valor numérico de confianza en la predicción (en este caso, no se utliza, aparece un .).
  - strand: hebra de ADn en la que se encuentra(+ o -).
  - frame: Marco de lectura (0, 1, 2) para las regiones codificantes, o . si no aplica.
  - gene_id: es el identificador único de este gen.
  - transcript_id: identificador del transcrito.
  - gene_name: el nombre de este gen, en este caso, no aplica, ya que en el archivo no se encuentra ninguno.

Adicionalmente, se adjuntan dos fotografías evidenciando el cambio generado al tranformar el formato del archivo; en la primera(datos_antes), se observa cómo se visualiza el archivo con el formato .gtf en Pycharm, y en la segunda(datos_después), cómo se ve cuando ya se transformó a formato .tsv.

## Acerca de la rutina

- Se encuentra en la parte superior como flatworm.py.
- Como se dijo anteriormente, esta rutina está pensada para ser utilizada en Python 3.
- Se debe descargar la base de datos y ubicarla en la carpeta y el ambiente virtual correspondiente.
- Se carga la rutina y se reemplazan los nombres de los archivos que sean necesarios.
- Una vez creado el archivo .tsv, este ya está listo para ser visualizado en Pycharm o en Excel.

# Segundo trabajo
## Primera parte
Para este trabajo se buscó una base de datos de pacientes que padecen de cáncer de ovario, y se utilizaron distintos tipos de gráficas para mostrar las relaciones que existen entre los datos recopilados, tanto cuantitativos como categóricos, en esta base de datos y cómo estas se pueden representar dependiendo de la gráfica utilizada.

- Estas gráficas se encuentran en el archivo adjuntado en Classroom.
- El archivo .tsv de la base de datos se adjunta en la parte superior de esta página.
- El código utilizado también se puede encontrar en la parte superior.

## Segunda parte 
En esta parte, se utiliza el código utilizado en clase para hallar el contenido de guanina-citosina de un genoma o secuencia de una proteína, en este caso, de *E.coli*, para hallar su valor-p a partir de una prueba estadística, y su gráfica correspondiente, que se encuentra en el archivo adjuntado a classroom.



  

  



