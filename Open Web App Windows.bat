@echo off
setlocal
cd /d "%~dp0"
set "URL=http://127.0.0.1:8765"
start "GE2340 Web App Server" /min python app\server.py
powershell -NoProfile -ExecutionPolicy Bypass -Command "$url = '%URL%'; $ready = $false; for ($i = 0; $i -lt 90; $i++) { try { $response = Invoke-WebRequest -Uri $url -TimeoutSec 2; if ($response.StatusCode -eq 200) { $ready = $true; break } } catch {}; Start-Sleep -Seconds 1 }; if ($ready) { Start-Process $url } else { Write-Host 'The server did not start. Check the GE2340 Web App Server window.'; pause }"
endlocal
