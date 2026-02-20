param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv",
    [double]$DemoDuration = 3.0,
    [double]$MeditationDuration = 60.0,
    [double]$CeremonialPreheat = 10.0,
    [double]$CeremonialDuration = 60.0,
    [switch]$SkipInstall,
    [switch]$SkipMeditation,
    [switch]$SkipCeremonial
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "[$stamp] $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments = @()
    )

    Write-Step $Name
    Write-Host ("CMD> {0} {1}" -f $FilePath, ($Arguments -join " "))

    & $FilePath @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw ("Step failed ({0}) with exit code {1}" -f $Name, $code)
    }
}

function Test-PythonImport {
    param(
        [string]$PythonExe,
        [string]$ModuleName
    )

    & $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)"
    return ($LASTEXITCODE -eq 0)
}

$startedAt = Get-Date
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
        throw ("Python executable not found: {0}" -f $Python)
    }

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Invoke-External -Name "Create virtual environment" -FilePath $Python -Arguments @("-m", "venv", $VenvPath)
    }

    if (-not (Test-Path $venvPython)) {
        throw ("Virtual environment python not found after creation: {0}" -f $venvPython)
    }

    $logDir = Join-Path $scriptDir "logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }
    $runId = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $logDir ("atheria_0_to_100_{0}.log" -f $runId)
    Start-Transcript -Path $logFile -Force | Out-Null

    if (-not $SkipInstall) {
        Invoke-External -Name "Upgrade pip/setuptools/wheel" -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
        Invoke-External -Name "Install/upgrade numpy" -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "numpy")
        Invoke-External -Name "Install/upgrade torch" -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "torch")
    }
    else {
        Write-Step "Skip install requested"
    }

    if (-not (Test-PythonImport -PythonExe $venvPython -ModuleName "torch")) {
        if ($SkipInstall) {
            throw "Dependency check failed: module 'torch' is missing in venv while -SkipInstall is set."
        }
        Invoke-External -Name "Install missing dependency torch" -FilePath $venvPython -Arguments @("-m", "pip", "install", "torch")
        if (-not (Test-PythonImport -PythonExe $venvPython -ModuleName "torch")) {
            throw "Dependency check failed: module 'torch' is still missing after installation."
        }
    }

    if (-not (Test-PythonImport -PythonExe $venvPython -ModuleName "numpy")) {
        if ($SkipInstall) {
            Write-Warning "Dependency hint: module 'numpy' is missing in venv. Torch may emit warnings."
        }
        else {
            Invoke-External -Name "Install missing dependency numpy" -FilePath $venvPython -Arguments @("-m", "pip", "install", "numpy")
            if (-not (Test-PythonImport -PythonExe $venvPython -ModuleName "numpy")) {
                throw "Dependency check failed: module 'numpy' is still missing after installation."
            }
        }
    }

    Invoke-External -Name "Compile Python files" -FilePath $venvPython -Arguments @("-m", "py_compile", "atheria_core.py", "main.py", "DEMO/forge_executable.py")
    Invoke-External -Name "CLI help" -FilePath $venvPython -Arguments @("main.py", "--help")
    Invoke-External -Name ("Run demo ({0}s)" -f $DemoDuration) -FilePath $venvPython -Arguments @("main.py", "demo", "--duration", "$DemoDuration")

    $snapshotPath = Join-Path $logDir ("morphic_snapshot_{0}.json" -f $runId)
    if (-not $SkipMeditation) {
        Invoke-External -Name ("Run meditation ({0}s)" -f $MeditationDuration) -FilePath $venvPython -Arguments @("main.py", "meditation", "--duration", "$MeditationDuration", "--snapshot", $snapshotPath)
    }
    else {
        Write-Step "Skip meditation requested"
    }

    if (-not $SkipCeremonial) {
        Invoke-External -Name ("Run ceremonial preheat={0}s duration={1}s" -f $CeremonialPreheat, $CeremonialDuration) -FilePath $venvPython -Arguments @("main.py", "ceremonial", "--preheat", "$CeremonialPreheat", "--duration", "$CeremonialDuration", "--snapshot", $snapshotPath)
    }
    else {
        Write-Step "Skip ceremonial requested"
    }

    Invoke-External -Name "Run full verification tests" -FilePath $venvPython -Arguments @(
        "-m", "unittest", "-v",
        "tests/test_digital_life_claims.py",
        "tests/test_claim_hardening.py",
        "tests/test_demo_forge.py",
        "tests/test_collective_extensions.py"
    )

    $elapsed = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 2)
    Write-Step ("0->100 pipeline finished successfully in {0}s" -f $elapsed)
    Write-Host ("Log file: {0}" -f $logFile) -ForegroundColor Green
    if ((Test-Path $snapshotPath) -and (-not $SkipMeditation -or -not $SkipCeremonial)) {
        Write-Host ("Snapshot: {0}" -f $snapshotPath) -ForegroundColor Green
    }
}
catch {
    Write-Error $_
    exit 1
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
    Pop-Location
}
