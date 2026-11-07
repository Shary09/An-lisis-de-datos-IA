@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo   Lab 01 - Analisis de datos
 echo ==============================================
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta agregado al PATH.
    echo Instala Python desde https://www.python.org/downloads/
    pause
    exit /b 1
  ) else (
    set "PYTHON_CMD=py"
  )
) else (
  set "PYTHON_CMD=python"
)
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
echo Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 goto error
echo Instalando dependencias. Esto puede tardar varios minutos...
pip install -r requirements.txt
if errorlevel 1 goto error
echo Iniciando la aplicacion...
streamlit run app_lab01.py
exit /b 0
:error
echo.
echo Ocurrio un error. Revisa los mensajes anteriores y el README.md.
pause
exit /b 1
