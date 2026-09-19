param(
    [int[]]$Days          = @(1, 2, 3, 7, 14),
    [int]$Patients        = 100,
    [int]$Seed            = 42,
    [int]$Restarts        = 25,
    [int]$Evals           = 1000,
    [int]$LmIters         = 30,
    [string]$ParamsConfig = "config/config_9_active.json",
    [double]$DailyChoG    = [double]::NaN,
    [int]$NMeals          = -1,
    [double]$SplitCv      = [double]::NaN,
    [double]$TimeCv       = [double]::NaN,
    [double]$BolusUPerG   = [double]::NaN,
    [array]$Modes         = @(
        @{Name="clean";           Preset="none";  ExtraFlags=""},
        @{Name="cgm10-meal10";    Preset="none";  ExtraFlags="--cgm-rel-sd 10 --meal-timing-shift 10"}
    )
)

$ScriptDir  = Split-Path -Parent $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = $PWD.Path }
$ProjectRoot = $ScriptDir
$ExePath  = Join-Path $ProjectRoot "build\crossval\cross_validate.exe"
$MingwBin = "C:\msys64\mingw64\bin"
$Launched = @()

# ---- 1. Ensure output directories exist ----
foreach ($mode in $Modes) {
    $outDir = Join-Path (Join-Path $ProjectRoot "crossval") "${Patients}-$($mode.Name)"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

# ---- 2. Build & launch ----
foreach ($mode in $Modes) {
    $outDir = Join-Path (Join-Path $ProjectRoot "crossval") "${Patients}-$($mode.Name)"

    foreach ($day in $Days) {
        $durationMin = $day * 1440
        $csvFile  = Join-Path $outDir "crossval_results_${day}d.csv"
        $jsonFile = Join-Path $outDir "crossval_results_${day}d.json"

        # ---- 2a. Skip if output files already exist ----
        if ((Test-Path $csvFile) -and (Test-Path $jsonFile)) {
            Write-Output "SKIP  [$($mode.Name), ${day}d] - output files exist"
            continue
        }

        # ---- 2b. Skip if a running process already writes to this CSV ----
        $alreadyRunning = $false
        $procs = Get-Process -Name "cross_validate" -ErrorAction SilentlyContinue
        foreach ($p in $procs) {
            try {
                $cmdLine = $p.CommandLine
                if ($cmdLine -match [regex]::Escape($csvFile)) {
                    Write-Output "SKIP  [$($mode.Name), ${day}d] - already running (PID $($p.Id))"
                    $alreadyRunning = $true
                    break
                }
            } catch { }
        }
        if ($alreadyRunning) { continue }

        # ---- 2c. Build argument list ----
        $argList = @(
            "--patients", $Patients,
            "--seed", $Seed,
            "--noise-preset", $mode.Preset
        )
        if ($mode.ExtraFlags) {
            $argList += $mode.ExtraFlags -split ' '
        }
        $argList += @(
            "--params-config", $ParamsConfig,
            "--duration-min", $durationMin,
            "--restarts", $Restarts,
            "--evals", $Evals,
            "--lm-iters", $LmIters,
            "--csv", $csvFile,
            "--json", $jsonFile
        )
        # Meal generation flags (if set)
        if ($DailyChoG -eq $DailyChoG)   { $argList += "--daily-cho-g";  $argList += $DailyChoG }
        if ($NMeals -gt 0)                { $argList += "--n-meals";     $argList += $NMeals }
        if ($SplitCv -eq $SplitCv)       { $argList += "--split-cv";    $argList += $SplitCv }
        if ($TimeCv -eq $TimeCv)         { $argList += "--time-cv";     $argList += $TimeCv }
        if ($BolusUPerG -eq $BolusUPerG) { $argList += "--bolus-u-per-g"; $argList += $BolusUPerG }

        # ---- 2d. Launch in a separate PowerShell window ----
        $psCmd = "`$env:Path = '$MingwBin;' + `$env:Path; " +
                 "Write-Host '=== [$($mode.Name), ${day}d] STARTING ===' -ForegroundColor Green; " +
                 "& '$ExePath' $($argList -join ' '); " +
                 "Write-Host '=== [$($mode.Name), ${day}d] DONE ===' -ForegroundColor Green; " +
                 "Read-Host 'Press Enter to close'"

        $proc = Start-Process -WindowStyle Normal -FilePath "powershell.exe" `
            -ArgumentList @("-NoExit", "-NoProfile", "-Command", $psCmd) -PassThru

        $Launched += @{
            Mode = $mode.Name
            Days = $day
            PID  = $proc.Id
            CSV  = $csvFile
            JSON = $jsonFile
        }

        Write-Output "LAUNCH [$($mode.Name), ${day}d] - PID $($proc.Id)"

        Start-Sleep -Milliseconds 600
    }
}

# ---- 3. Summary ----
Write-Output "`n========================================"
Write-Output "  All launches submitted"
Write-Output "  Launched: $($Launched.Count)"
Write-Output "========================================"
foreach ($item in $Launched) {
    $msg = "  PID " + $item.PID + "  [" + $item.Mode + ", " + $item.Days + "d]"
    Write-Output $msg
}
