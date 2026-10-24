@echo off
REM Lanza las 8 combinaciones de XGBoost (4 balanceos x Grid/Random) en paralelo,
REM cada una en su propia ventana. Ejecutar desde el Anaconda Prompt con el ambiente
REM del proyecto ya activado, parado en la carpeta proyecto-final-covertype.
cd /d "%~dp0"
if not exist resultados\xgb_espacio_completo mkdir resultados\xgb_espacio_completo
for %%B in (smote adasyn balanced ninguno) do (
  for %%O in (grid random) do (
    start "XGB %%B %%O" cmd /k "python scripts\xgb_espacio_completo.py --balanceo %%B --opt %%O > resultados\xgb_espacio_completo\%%B_%%O.log 2>&1 & type resultados\xgb_espacio_completo\%%B_%%O.log"
  )
)
echo Se abrieron 8 ventanas. Cada una se puede cerrar cuando termine.
