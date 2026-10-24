# Lanza en paralelo las combinaciones de resultados/combinaciones_a_repetir.txt,
# un proceso por combinacion y un hilo por proceso (ver scripts/uniformizar_protocolo.py).
# Uso:  powershell -File scripts\lanzar_repeticion.ps1 [-Python <ruta a python.exe>]
param([string]$Python = "python")
$raiz = Split-Path -Parent $PSScriptRoot
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$logs = Join-Path $raiz "resultados\repeticion\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
Get-Content (Join-Path $raiz "resultados\combinaciones_a_repetir.txt") | Where-Object { $_.Trim() } | ForEach-Object {
    $nombre = $_ -replace '\|', '_'
    Start-Process -FilePath $Python -ArgumentList "scripts\run_benchmark.py", "--combo", "`"$_`"" `
        -WorkingDirectory $raiz -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "$nombre.out") -RedirectStandardError (Join-Path $logs "$nombre.err")
}
Write-Output "Procesos lanzados."
