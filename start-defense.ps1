param(
    [switch]$Prepare,
    [switch]$RebuildDashboard,
    [switch]$NoBrowser,
    [switch]$Stop,
    [string]$SerialPort = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$RuntimeDir = Join-Path $Root ".defense"
$PidFile = Join-Path $RuntimeDir "pids.json"
$LogDir = Join-Path $RuntimeDir "logs"
$DatabaseUrl = "postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"
$env:TINYML_DATABASE_URL = $DatabaseUrl

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step([string]$Text) {
    Write-Host "`n==> $Text" -ForegroundColor Cyan
}

function Write-Ok([string]$Text) {
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Write-Warn([string]$Text) {
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMs = 1000) {
    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $client) { $client.Close() }
    }
}

function Stop-OwnedProcesses {
    if (-not (Test-Path $PidFile)) { return }

    try {
        $data = Get-Content $PidFile -Raw | ConvertFrom-Json
        foreach ($prop in $data.PSObject.Properties) {
            if ($null -eq $prop.Value) { continue }
            $pidValue = [int]$prop.Value
            $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($null -ne $proc) {
                Write-Host "Stopping $($prop.Name) (PID $pidValue)..."
                Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            }
        }
    }
    catch {
        Write-Warn "Could not read old PID file: $($_.Exception.Message)"
    }

    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

if ($Stop) {
    Write-Step "Stopping defense launcher processes"
    Stop-OwnedProcesses
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        try {
            docker compose stop postgres | Out-Host
        }
        catch {
            Write-Warn "PostgreSQL container could not be stopped automatically."
        }
    }
    Write-Ok "Defense stack stopped."
    exit 0
}

# Clean only processes previously started by THIS launcher.
Stop-OwnedProcesses
$OwnedPids = [ordered]@{}
function Save-Pids {
    $OwnedPids | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8
}

Write-Step "Checking repository"
$RequiredPaths = @(
    "requirements.txt",
    "docker-compose.yml",
    "server\app\r1_mqtt.py",
    "server\app\main.py",
    "dashboard\package.json"
)
foreach ($path in $RequiredPaths) {
    if (-not (Test-Path (Join-Path $Root $path))) {
        throw "Required project file not found: $path`nPut start-defense.ps1 in the repository root."
    }
}
Write-Ok "Repository root: $Root"

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if ($Prepare) {
    Write-Step "Preparing Python environment"
    if (-not (Test-Path $VenvPython)) {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $py) {
            & $py.Source -3 -m venv .venv
        }
        else {
            $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($null -eq $pythonCmd) { throw "Python was not found in PATH." }
            & $pythonCmd.Source -m venv .venv
        }
    }
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    Write-Ok "Python dependencies ready."

    Write-Step "Preparing dashboard dependencies and production build"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($null -eq $npm) { throw "npm was not found in PATH. Install Node.js first." }
    Push-Location (Join-Path $Root "dashboard")
    try {
        & $npm.Source install
        if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) { throw "Dashboard build failed." }
    }
    finally {
        Pop-Location
    }
    Write-Ok "Dashboard production build ready."
}

if (-not (Test-Path $VenvPython)) {
    throw "Python virtual environment is missing. Run once:`n  powershell -ExecutionPolicy Bypass -File .\start-defense.ps1 -Prepare"
}

$DashboardDist = Join-Path $Root "dashboard\dist"
if ($RebuildDashboard -or -not (Test-Path $DashboardDist)) {
    Write-Step "Building dashboard"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($null -eq $npm) { throw "npm was not found in PATH." }
    $NodeModules = Join-Path $Root "dashboard\node_modules"
    if (-not (Test-Path $NodeModules)) {
        throw "dashboard/node_modules is missing. Run the launcher once with -Prepare."
    }
    Push-Location (Join-Path $Root "dashboard")
    try {
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) { throw "Dashboard build failed." }
    }
    finally {
        Pop-Location
    }
    Write-Ok "Dashboard build completed."
}
else {
    Write-Ok "Existing dashboard/dist will be used."
}

Write-Step "Checking Docker / PostgreSQL"
$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($null -eq $docker) {
    throw "Docker CLI was not found. Install/start Docker Desktop before the defense."
}

function Test-DockerEngineReady {
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $docker.Source
        $psi.Arguments = 'version --format "{{.Server.Version}}"'
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true

        $proc = New-Object System.Diagnostics.Process
        $proc.StartInfo = $psi
        [void]$proc.Start()
        $stdout = $proc.StandardOutput.ReadToEnd()
        $stderr = $proc.StandardError.ReadToEnd()
        $proc.WaitForExit()

        return ($proc.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($stdout))
    }
    catch {
        return $false
    }
}

$dockerReady = Test-DockerEngineReady

if (-not $dockerReady) {
    $dockerDesktopCandidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
    )
    $dockerDesktop = $dockerDesktopCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($null -eq $dockerDesktop) {
        throw "Docker engine is not running and Docker Desktop could not be located. Start Docker Desktop and run this script again."
    }

    Write-Warn "Docker engine is not running; starting Docker Desktop..."
    Start-Process -FilePath $dockerDesktop | Out-Null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        if (Test-DockerEngineReady) {
            $dockerReady = $true
            break
        }
    }
    if (-not $dockerReady) { throw "Docker did not become ready within 120 seconds." }
}
Write-Ok "Docker engine is ready."

& $docker.Source compose up -d postgres | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Could not start PostgreSQL container." }

$postgresReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        & $docker.Source compose exec -T postgres pg_isready -U tinyml -d tinyml *> $null
        if ($LASTEXITCODE -eq 0) { $postgresReady = $true; break }
    }
    catch {}
}
if (-not $postgresReady) { throw "PostgreSQL did not become healthy." }
Write-Ok "PostgreSQL is healthy on 127.0.0.1:5432."

Write-Step "Checking MQTT broker"
if (-not (Test-TcpPort "127.0.0.1" 1883)) {
    $mosquittoService = Get-Service -Name "mosquitto" -ErrorAction SilentlyContinue
    if ($null -ne $mosquittoService) {
        if ($mosquittoService.Status -ne "Running") {
            Start-Service -Name "mosquitto"
            Start-Sleep -Seconds 2
        }
    }
    else {
        $mosquitto = Get-Command mosquitto -ErrorAction SilentlyContinue
        if ($null -eq $mosquitto) {
            throw "MQTT broker is not reachable on port 1883 and Mosquitto was not found. Start Mosquitto before running the defense stack."
        }
        $mosqOut = Join-Path $LogDir "mosquitto.out.log"
        $mosqErr = Join-Path $LogDir "mosquitto.err.log"
        $mosqProcess = Start-Process -FilePath $mosquitto.Source -ArgumentList @("-v") -PassThru -RedirectStandardOutput $mosqOut -RedirectStandardError $mosqErr -WindowStyle Hidden
        $OwnedPids["mosquitto"] = $mosqProcess.Id
        Save-Pids
        Start-Sleep -Seconds 2
    }
}
if (-not (Test-TcpPort "127.0.0.1" 1883)) {
    throw "MQTT broker is still not reachable on 127.0.0.1:1883."
}
Write-Ok "MQTT broker is reachable on port 1883."

# Production firmware currently targets this Windows Mobile Hotspot address.
$HotspotIp = Get-NetIPAddress -IPAddress "192.168.137.1" -ErrorAction SilentlyContinue
if ($null -eq $HotspotIp) {
    Write-Warn "192.168.137.1 is not present on this PC. The production firmware currently targets broker 192.168.137.1:1883. Turn ON Windows Mobile Hotspot before the live ESP32 demo."
}
else {
    Write-Ok "Defense hotspot address 192.168.137.1 is present."
}

if (Test-TcpPort "127.0.0.1" 8000) {
    throw "Port 8000 is already in use. Close the old dashboard/API process, then run this script again."
}

Write-Step "Starting R1 MQTT inference + persistence service"
$R1Out = Join-Path $LogDir "r1_mqtt.out.log"
$R1Err = Join-Path $LogDir "r1_mqtt.err.log"
Remove-Item $R1Out, $R1Err -Force -ErrorAction SilentlyContinue
$R1Process = Start-Process -FilePath $VenvPython -ArgumentList @(
    "-u", "-m", "server.app.r1_mqtt",
    "--database-url", $DatabaseUrl
) -WorkingDirectory $Root -PassThru -RedirectStandardOutput $R1Out -RedirectStandardError $R1Err -WindowStyle Hidden
$OwnedPids["r1_mqtt"] = $R1Process.Id
Save-Pids

$R1Ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ($R1Process.HasExited) {
        $errText = if (Test-Path $R1Err) { Get-Content $R1Err -Raw } else { "" }
        throw "R1 MQTT service exited early.`n$errText"
    }
    if (Test-Path $R1Out) {
        $text = Get-Content $R1Out -Raw
        if ($text -match "R1_SERVER_READY") { $R1Ready = $true; break }
    }
}
if (-not $R1Ready) { throw "R1 MQTT service did not report R1_SERVER_READY. Check $R1Out and $R1Err" }
Write-Ok "R1 MQTT inference/persistence service is ready."

Write-Step "Starting FastAPI dashboard service"
$ApiOut = Join-Path $LogDir "dashboard_api.out.log"
$ApiErr = Join-Path $LogDir "dashboard_api.err.log"
Remove-Item $ApiOut, $ApiErr -Force -ErrorAction SilentlyContinue
$ApiProcess = Start-Process -FilePath $VenvPython -ArgumentList @(
    "-u", "-m", "server.app.main",
    "--database-url", $DatabaseUrl,
    "--host", "127.0.0.1",
    "--port", "8000"
) -WorkingDirectory $Root -PassThru -RedirectStandardOutput $ApiOut -RedirectStandardError $ApiErr -WindowStyle Hidden
$OwnedPids["dashboard_api"] = $ApiProcess.Id
Save-Pids

$Health = $null
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    if ($ApiProcess.HasExited) {
        $errText = if (Test-Path $ApiErr) { Get-Content $ApiErr -Raw } else { "" }
        throw "Dashboard API exited early.`n$errText"
    }
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/dashboard/health" -TimeoutSec 3
        if ($Health.status -eq "ok") { break }
    }
    catch {}
}
if ($null -eq $Health -or $Health.status -ne "ok") {
    throw "Dashboard API did not become healthy. Check $ApiOut and $ApiErr"
}
Write-Ok "Dashboard API is healthy."

if ($SerialPort -ne "") {
    Write-Step "Starting optional ESP32 serial monitor on $SerialPort"
    $pioPath = Join-Path $Root ".venv\Scripts\pio.exe"
    if (-not (Test-Path $pioPath)) {
        $pio = Get-Command pio -ErrorAction SilentlyContinue
        if ($null -ne $pio) { $pioPath = $pio.Source }
    }

    if (Test-Path $pioPath) {
        $SerialProcess = Start-Process -FilePath $pioPath -ArgumentList @(
            "device", "monitor", "--port", $SerialPort, "--baud", "115200"
        ) -WorkingDirectory (Join-Path $Root "firmware") -PassThru
        $OwnedPids["serial_monitor"] = $SerialProcess.Id
        Save-Pids
        Write-Ok "Serial monitor started."
    }
    else {
        Write-Warn "PlatformIO CLI was not found; serial monitor was skipped."
    }
}

$DashboardUrl = "http://127.0.0.1:8000"
if (-not $NoBrowser) {
    Start-Process $DashboardUrl | Out-Null
}

Write-Host "`n============================================================" -ForegroundColor DarkCyan
Write-Host "  ADAPTIVE EDGE-CLOUD TINYML - DEFENSE STACK READY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "Dashboard : $DashboardUrl"
Write-Host "API health: http://127.0.0.1:8000/api/dashboard/health"
Write-Host "MQTT      : 127.0.0.1:1883"
Write-Host "PostgreSQL: 127.0.0.1:5432 / tinyml"
Write-Host "Events    : $($Health.event_count)"
Write-Host "Pose rows : $($Health.pose_event_count)"
Write-Host "Logs      : $LogDir"
Write-Host ""
Write-Host "For the live board, keep Windows Mobile Hotspot ON." -ForegroundColor Yellow
Write-Host "The current production firmware targets 192.168.137.1:1883." -ForegroundColor Yellow
Write-Host ""
Write-Host "Stop this launcher later with:" -ForegroundColor Gray
Write-Host "  powershell -ExecutionPolicy Bypass -File .\start-defense.ps1 -Stop" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host ""
