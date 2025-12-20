#!/sbin/bash

read -p "Introduce el directorio que contiene archivos .heic o .HEIC: " DIR

echo "Buscando archivos en: $DIR"

# Verificamos si el directorio existe
if [[ ! -d "$DIR" ]]; then
    echo "El directorio no existe."
    exit 1
fi

# Buscamos los archivos .heic o .HEIC (ignorando mayúsculas)
shopt -s nocaseglob
for FILE in "$DIR"/*.heic "$DIR"/*.HEIC; do
    # Verificamos que el archivo exista (por si no hay coincidencias)
    if [[ -f "$FILE" ]]; then
        # Creamos nombre de salida .jpg
        OUT_FILE="${FILE%.*}.jpg"
        echo "Convirtiendo: $FILE → $OUT_FILE"
        heif-convert "$FILE" "$OUT_FILE"
    else
        echo "No se encontraron archivos .heic/.HEIC en $DIR"
    fi
done
shopt -u nocaseglob
